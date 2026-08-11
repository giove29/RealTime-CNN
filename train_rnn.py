"""
Addestra lo strato temporale che legge la sequenza di predizioni della CNN.

CONTESTO: la CNN classifica un secondo di audio alla volta. Alcuni eventi pero'
non esistono dentro un secondo: un incidente e' una sequenza, e un fuoco
d'artificio si distingue da uno sparo solo per la ripetizione. Questo modello
legge quindici predizioni consecutive (7,5 secondi) e riconosce lo scenario.

DUE ARCHITETTURE A CONFRONTO:

  --tipo gru   Rete ricorrente. E' la scelta "naturale" per una sequenza, ma
               su TFLite Micro gli operatori ricorrenti sono molto piu' esotici
               di DepthwiseConv2D, che sul Pico si e' rivelato bloccante. Serve
               qui come termine di paragone, non per il deployment.

  --tipo cnn   Convoluzione sull'asse temporale. La sequenza ha lunghezza fissa
               (15 passi), quindi la ricorrenza non e' necessaria: una
               convoluzione vede tutta la finestra. Usa solo CONV_2D,
               AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED e SOFTMAX, cioe' gli
               stessi operatori gia' verificati sul tuo dispositivo.

Uso:
    python train_rnn.py --confronta          # addestra entrambe e confronta
    python train_rnn.py --tipo cnn           # addestra e quantizza la CNN
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET = "rnn_dataset.npz"
MODEL_OUT = "temporal_model.h5"
TFLITE_OUT = "temporal_int8.tflite"
CPP_OUT = "model_temporal_data.cc"


def costruisci_gru(n_passi, n_classi_cnn, n_scenari):
    inp = layers.Input(shape=(n_passi, n_classi_cnn))
    x = layers.GRU(32, return_sequences=False)(inp)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_scenari, activation='softmax')(x)
    return models.Model(inp, out)


def costruisci_cnn(n_passi, n_classi_cnn, n_scenari):
    """Ingresso trattato come immagine (passi x classi x 1).

    Il primo kernel e' (3, n_classi): copre tre passi temporali e TUTTE le
    classi insieme, cioe' guarda 1,5 secondi di predizioni per volta. Il
    secondo scorre solo nel tempo. La media finale rende la decisione
    indipendente da DOVE cade l'evento nella finestra.
    """
    inp = layers.Input(shape=(n_passi, n_classi_cnn, 1))
    x = layers.Conv2D(24, (3, n_classi_cnn), padding='valid', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (3, 1), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.AveragePooling2D(pool_size=(h, w))(x)
    x = layers.Reshape((c,))(x)          # forma esplicita: Flatten genera
    x = layers.Dropout(0.3)(x)           # SHAPE/STRIDED_SLICE/PACK
    out = layers.Dense(n_scenari, activation='softmax')(x)
    return models.Model(inp, out)


def addestra(model, Xtr, ytr, Xte, yte, epoche=120):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epoche,
                  batch_size=32, verbose=2, callbacks=[
                      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=8, verbose=1, min_lr=1e-6),
                      callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                              restore_best_weights=True, verbose=1)])
    return max(h.history['val_accuracy'])


def quantizza(model, Xrep, scenari):
    """Stessa procedura di quantize_model.py: batch fissa a 1, calibrazione su
    dati reali, controllo degli operatori prima di generare il file C."""
    forma = tuple(model.inputs[0].shape[1:])
    fissa = tf.keras.models.clone_model(
        model, input_tensors=tf.keras.Input(batch_shape=(1,) + forma))
    fissa.set_weights(model.get_weights())

    conv = tf.lite.TFLiteConverter.from_keras_model(fissa)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rappresentativo():
        for i in range(min(200, len(Xrep))):
            yield [np.expand_dims(Xrep[i], 0).astype(np.float32)]

    conv.representative_dataset = rappresentativo
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()

    with open(TFLITE_OUT, 'wb') as f:
        f.write(blob)
    print(f"\nModello quantizzato: {len(blob)/1024:.2f} KB")

    # Solo operatori gia' presenti nel resolver del firmware: il modello
    # temporale condivide il MicroMutableOpResolver con la CNN principale.
    ATTESI = {"CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED",
              "SOFTMAX", "QUANTIZE", "DEQUANTIZE", "MAX_POOL_2D"}
    IGNORA = {"DELEGATE"}
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    print("\nOperatori:")
    problemi = []
    for d in it._get_ops_details():
        if d['op_name'] in IGNORA:
            continue
        ok = d['op_name'] in ATTESI
        print(f"  {'OK ' if ok else '>>>'} {d['op_name']}")
        if not ok:
            problemi.append(d['op_name'])
    if problemi:
        print(f"\nATTENZIONE: operatori non nel resolver del firmware -> "
              f"{sorted(set(problemi))}\nNon flashare finche' non sono gestiti.")
    else:
        print("\nTutti gli operatori sono gia' supportati dal firmware.")

    inp = it.get_input_details()[0]
    print(f"Input: {tuple(inp['shape'])}  scale={inp['quantization'][0]:.8f} "
          f"zp={inp['quantization'][1]}")

    hexs = [f"0x{b:02x}" for b in blob]
    righe = [", ".join(hexs[i:i+12]) for i in range(0, len(hexs), 12)]
    with open(CPP_OUT, 'w') as f:
        f.write("// Modello temporale quantizzato - generato da train_rnn.py\n"
                "#include <cstdint>\n\n"
                "// Scenari, nell'ordine degli indici di uscita:\n")
        for i, s in enumerate(scenari):
            f.write(f"//   {i} = {s}\n")
        f.write("\nalignas(8) extern const unsigned char g_temporal_model[] = {\n  "
                + ",\n  ".join(righe) +
                f"\n}};\n\nextern const unsigned int g_temporal_model_len = {len(blob)};\n")
    print(f"Scritto '{CPP_OUT}'. Copialo nel progetto Pico.")

    # L'ordine degli scenari e' l'unico punto in cui un disallineamento non
    # produce errori ma solo nomi sbagliati. Meglio darlo gia' pronto.
    print("\n" + "=" * 62)
    print("Incolla questo in main.cpp al posto dell'array SCENARI esistente:")
    print("=" * 62)
    print("static const char* SCENARI[] = {")
    for i in range(0, len(scenari), 3):
        print("    " + ", ".join(f'"{s}"' for s in scenari[i:i+3]) + ",")
    print("};")
    print(f"#define NUM_SCENARI {len(scenari)}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=("cnn", "gru"), default="cnn")
    ap.add_argument("--confronta", action="store_true",
                    help="addestra entrambe e mostra il confronto")
    ap.add_argument("--epoche", type=int, default=120)
    args = ap.parse_args()

    if not os.path.exists(DATASET):
        sys.exit(f"Manca '{DATASET}'. Esegui prima make_rnn_dataset.py.")
    d = np.load(DATASET, allow_pickle=True)
    X, y = d['X'], d['y']
    scenari = [str(s) for s in d['scenari']]
    n_passi, n_cls = X.shape[1], X.shape[2]

    print(f"Sequenze: {X.shape}   scenari: {', '.join(scenari)}")
    for i, s in enumerate(scenari):
        print(f"  {s:<20} {int((y == i).sum()):5d}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

    if args.confronta:
        print("\n=== GRU ===")
        g = costruisci_gru(n_passi, n_cls, len(scenari))
        g.summary()
        acc_g = addestra(g, Xtr, ytr, Xte, yte, args.epoche)

        print("\n=== CNN temporale ===")
        c = costruisci_cnn(n_passi, n_cls, len(scenari))
        c.summary()
        acc_c = addestra(c, Xtr[..., np.newaxis], ytr,
                         Xte[..., np.newaxis], yte, args.epoche)

        print(f"\nGRU {acc_g:.4f}   CNN temporale {acc_c:.4f}")
        print("Se la differenza e' sotto i due punti, conviene la CNN: usa solo\n"
              "operatori gia' verificati sul Pico, la GRU no.")
        return

    if args.tipo == "gru":
        model = costruisci_gru(n_passi, n_cls, len(scenari))
        Xtr_, Xte_ = Xtr, Xte
    else:
        model = costruisci_cnn(n_passi, n_cls, len(scenari))
        Xtr_, Xte_ = Xtr[..., np.newaxis], Xte[..., np.newaxis]

    model.summary()
    acc = addestra(model, Xtr_, ytr, Xte_, yte, args.epoche)
    print(f"\nMigliore val_accuracy: {acc:.4f}")

    pred = model.predict(Xte_, verbose=0).argmax(axis=1)
    print("\n" + classification_report(yte, pred, target_names=scenari, digits=3))
    print("Matrice di confusione (righe = vere, colonne = predette):")
    cm = confusion_matrix(yte, pred)
    print("            " + " ".join(f"{s[:7]:>8}" for s in scenari))
    for i, s in enumerate(scenari):
        print(f"{s[:11]:<11} " + " ".join(f"{v:8d}" for v in cm[i]))

    model.save(MODEL_OUT)
    print(f"\nSalvato '{MODEL_OUT}'.")

    if args.tipo == "gru":
        print("\nLa GRU non viene quantizzata: gli operatori ricorrenti non sono\n"
              "nel resolver del firmware. Usa --tipo cnn per il deployment.")
    else:
        quantizza(model, Xtr_, scenari)


if __name__ == "__main__":
    main()
