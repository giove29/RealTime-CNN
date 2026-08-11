"""
Fine-tuning del modello sulle registrazioni fatte con il Pico.

Il modello addestrato sui file del dataset raggiunge l'80-90% di confidenza sul
PC, ma sul dispositivo la stessa registrazione ne fa 45%. Il divario nasce dalla
catena di acquisizione - microfono, ADC a 12 bit, filtro di decimazione - non
dal contenuto dei suoni. Riaddestrare gli ULTIMI layer su registrazioni fatte
col Pico riallinea il modello a quel dominio senza fargli dimenticare quello
che ha imparato dal dataset grande.

Si congelano i primi blocchi perche' estraggono caratteristiche generiche
(bordi, transienti, bande) valide in entrambi i domini. Gli ultimi layer sono
quelli che decidono, ed e' li' che il divario si manifesta.

Uso:
    python fine_tune.py registrazioni_auto
    python fine_tune.py registrazioni_auto --scongela 2 --epoche 40
    python fine_tune.py registrazioni_auto --mescola dataset_features.npz
"""
import argparse
import os
import sys

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- devono coincidere con extract_features.py ---
SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 32
HOP_LENGTH = 256
N_FFT = 512
MIN_DB, MAX_DB = -80.0, 20.0

MODEL_IN = "smartcity_model.h5"
MODEL_OUT = "smartcity_model_ft.h5"
CALIBRATION_OUT = "calibration_data.npy"

CATEGORIE = [
    "Attività_Umana", "Ambiente_Urbano", "Veicoli", "Sirene_e_Urla",
    "Spari", "Incidente", "Vetri", "Fuochi",
]


def feature(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, center=False)
    lm = librosa.power_to_db(mel, ref=1.0, top_db=None)
    lm = np.clip(lm, MIN_DB, MAX_DB)
    return ((lm - MIN_DB) / (MAX_DB - MIN_DB))[..., np.newaxis].astype(np.float32)


def carica(cartella):
    n_win = int(SAMPLE_RATE * DURATION)
    X, y = [], []
    for idx, classe in enumerate(CATEGORIE):
        d = os.path.join(cartella, classe)
        if not os.path.isdir(d):
            print(f"  {classe:<20}     0  (cartella assente)")
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith((".wav", ".flac"))]
        n = 0
        for f in files:
            try:
                a, _ = librosa.load(os.path.join(d, f), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue
            a = a[:n_win] if a.size >= n_win else np.pad(a, (0, n_win - a.size))
            if np.max(np.abs(a)) < 1e-4:      # clip praticamente muta
                continue
            X.append(feature(a))
            y.append(idx)
            n += 1
        print(f"  {classe:<20} {n:5d}")
    if not X:
        sys.exit("Nessuna registrazione caricata. Controlla i nomi delle cartelle.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cartella", help="cartella con una sottocartella per classe")
    ap.add_argument("--scongela", type=int, default=2,
                    help="quanti layer con pesi lasciare addestrabili, dal fondo "
                         "(1 = solo Dense, 2 = ultima conv + Dense)")
    ap.add_argument("--epoche", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate basso: si adatta, non si riparte da zero")
    ap.add_argument("--mescola", default=None,
                    help="npz del dataset originale da mescolare, per non "
                         "dimenticare cio' che si era imparato")
    ap.add_argument("--quota-originale", type=float, default=0.5,
                    help="quanti campioni originali per ogni registrazione Pico")
    args = ap.parse_args()

    print("Registrazioni caricate:")
    X, y = carica(args.cartella)
    print(f"  totale {len(X)}   forma {X[0].shape}\n")

    if not os.path.exists(MODEL_IN):
        sys.exit(f"Manca '{MODEL_IN}'.")
    model = tf.keras.models.load_model(MODEL_IN)
    if tuple(model.input_shape[1:]) != X[0].shape:
        sys.exit(f"Il modello vuole {model.input_shape[1:]}, le feature sono "
                 f"{X[0].shape}. Controlla i parametri in cima allo script.")

    # Congela tutto tranne gli ultimi layer che hanno pesi. BatchNormalization
    # resta congelata comunque: con pochi dati le sue statistiche mobili si
    # rovinerebbero, ed e' un errore classico del fine-tuning.
    con_pesi = [l for l in model.layers if l.trainable_weights]
    da_addestrare = con_pesi[-args.scongela:]
    for l in model.layers:
        l.trainable = False
    for l in da_addestrare:
        if not isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = True
    print("Layer addestrabili:",
          [l.name for l in model.layers if l.trainable] or "nessuno")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # Mescolare campioni originali evita l'oblio catastrofico: con solo dati
    # del Pico il modello si specializza e perde generalita'.
    if args.mescola:
        d = np.load(args.mescola)
        Xo, yo = d["X"], d["y"]
        n = min(len(Xo), int(len(X_tr) * args.quota_originale))
        sel = np.random.default_rng(0).choice(len(Xo), n, replace=False)
        X_tr = np.concatenate([X_tr, Xo[sel]])
        y_tr = np.concatenate([y_tr, yo[sel]])
        print(f"Mescolati {n} campioni originali: training su {len(X_tr)}")

    print("\nPrestazione PRIMA del fine-tuning, sulle registrazioni Pico:")
    p0 = model.predict(X_te, verbose=0).argmax(axis=1)
    acc0 = float((p0 == y_te).mean())
    print(f"  accuratezza {acc0:.3f}")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=args.epoche, batch_size=32, verbose=2,
              callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                 restore_best_weights=True, verbose=1)])

    p1 = model.predict(X_te, verbose=0).argmax(axis=1)
    acc1 = float((p1 == y_te).mean())
    print(f"\nAccuratezza sulle registrazioni Pico: {acc0:.3f} -> {acc1:.3f}")
    print("\n" + classification_report(y_te, p1, target_names=CATEGORIE,
                                       digits=3, zero_division=0))
    print("Matrice di confusione (righe = vere, colonne = predette):")
    cm = confusion_matrix(y_te, p1, labels=range(len(CATEGORIE)))
    print("            " + " ".join(f"{n[:6]:>7}" for n in CATEGORIE))
    for i, n in enumerate(CATEGORIE):
        print(f"{n[:11]:<11} " + " ".join(f"{v:7d}" for v in cm[i]))

    model.save(MODEL_OUT)
    # La calibrazione per la quantizzazione deve venire dal dominio del Pico,
    # altrimenti scale e zero point sono tarati sulla distribuzione sbagliata.
    np.save(CALIBRATION_OUT, X_tr[:100])
    print(f"\nSalvato '{MODEL_OUT}' e nuovi dati di calibrazione.")
    print(f"Per portarlo sul Pico: rinomina '{MODEL_OUT}' in '{MODEL_IN}' "
          f"(tieni una copia dell'originale) ed esegui quantize_model.py.")


if __name__ == "__main__":
    main()
