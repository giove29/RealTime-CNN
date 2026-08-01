import os
import numpy as np
import librosa
from tqdm import tqdm  # <-- Importazione della barra di avanzamento

# --- 1. CONFIGURAZIONE ---
DATASET_PATH = "dataset" # Cartella principale che contiene le tue 8 categorie
OUTPUT_FILE = "dataset_features.npz"

SAMPLE_RATE = 16000
DURATION = 2.0  # <-- Incrementato a 2 secondi per maggiore contesto!
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Parametri Log-Mel Spectrogram (Stato dell'arte per embedded)
N_MELS = 40
HOP_LENGTH = 512
N_FFT = 1024

# Limiti per la Normalizzazione Globale dei Decibel
MIN_DB = -80.0
MAX_DB = 20.0

# Le tue 8 categorie esatte
CATEGORIE = [
    "Attività_Umana",
    "Ambiente_Urbano",
    "Veicoli",
    "Sirene_e_Urla",
    "Spari",
    "Incidente",
    "Vetri",
    "Fuochi"
]

def process_audio(file_path):
    """
    Carica l'audio (WAV o MP3), lo porta a 16kHz mono, 
    ne fissa la durata a 2 secondi e ne estrae il Log-Mel Spectrogramma.
    """
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # 1. Taglio o Padding per avere esattamente 2.0 secondi
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    elif len(audio) < SAMPLES_PER_TRACK:
        audio = np.pad(audio, (0, SAMPLES_PER_TRACK - len(audio)), 'constant')
        
    # 2. Creazione dello Spettrogramma di potenza (Mel Scale)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    
    # 3. Conversione in Decibel con riferimento assoluto (ref=1.0)
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    
    # 4. Normalizzazione Globale (Mantiene i volumi reali)
    log_mel = np.clip(log_mel, MIN_DB, MAX_DB)
    log_mel = (log_mel - MIN_DB) / (MAX_DB - MIN_DB)
    
    # 5. Aggiunta della dimensione del canale (TensorFlow richiede X, Y, Canali)
    return log_mel[..., np.newaxis]

if __name__ == "__main__":
    print("Inizio estrazione feature allo stato dell'arte (Finestra: 2.0 secondi)...\n")
    
    X, y = [], []
    file_processati = 0
    file_scartati = 0
    
    for label_idx, folder in enumerate(CATEGORIE):
        folder_path = os.path.join(DATASET_PATH, folder)
        
        if not os.path.isdir(folder_path):
            print(f"⚠️  ATTENZIONE: Cartella mancante -> {folder}")
            continue
            
        # Filtra in anticipo solo i file audio per dare a tqdm un conteggio accurato
        file_audio_validi = [f for f in os.listdir(folder_path) if f.lower().endswith((".wav", ".mp3"))]
        
        if not file_audio_validi:
            print(f"Nessun file audio trovato nella cartella: {folder}")
            continue
        
        # Inizializza la barra di avanzamento per la categoria corrente
        for filename in tqdm(file_audio_validi, desc=f"Elaborando {folder: <16}", unit="file"):
            filepath = os.path.join(folder_path, filename)
            try:
                features = process_audio(filepath)
                X.append(features)
                y.append(label_idx)
                file_processati += 1
            except Exception as e:
                # Usiamo tqdm.write al posto di print per non spezzare visivamente la barra
                tqdm.write(f"   [!] Errore nel file {filename}: {e}")
                file_scartati += 1

    # Conversione in matrici Numpy ottimizzate per il training
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    classes = np.array(CATEGORIE)

    # Salvataggio del blocco dati
    np.savez(OUTPUT_FILE, X=X, y=y, classes=classes)
    
    print("\n" + "="*40)
    print("ESTRAZIONE COMPLETATA CON SUCCESSO!")
    print(f"File processati validi: {file_processati}")
    print(f"File saltati/corrotti:  {file_scartati}")
    print(f"Dimensioni matrice X:   {X.shape} (File, Mel-Bands, Time-Frames, Canale)")
    print(f"File salvato:           {OUTPUT_FILE}")
    print("="*40)