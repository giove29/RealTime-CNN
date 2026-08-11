import os
import numpy as np
import librosa
from tqdm import tqdm

# --- 1. CONFIGURAZIONE ---
DATASET_PATH = "dataset" 
OUTPUT_FILE = "dataset_features.npz"

SAMPLE_RATE = 16000
DURATION = 1.0  # Finestra piu' corta: gli eventi impulsivi (spari, vetri)
                # non hanno bisogno di contesto lungo, e si dimezza la latenza
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Parametri Log-Mel Spectrogram (Ottimizzati per Edge AI)
N_MELS = 32     # Piu' risoluzione in frequenza: serve a separare spari,
                # vetri e fuochi, che si distinguono sopra i 2 kHz
HOP_LENGTH = 256  # Meta' di N_FFT: frame sovrapposti al 50%, risoluzione
                  # temporale raddoppiata. Un transiente da 10 ms ora cade in
                  # due o tre colonne invece che in una sola.
N_FFT = 512

MIN_DB = -80.0
MAX_DB = 20.0

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
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    elif len(audio) < SAMPLES_PER_TRACK:
        audio = np.pad(audio, (0, SAMPLES_PER_TRACK - len(audio)), 'constant')
        
    # Normalizzazione di picco: senza questa lo spettrogramma del microfono
    # sul Pico risulta traslato di decine di dB rispetto ai file del dataset,
    # e la CNN vede una distribuzione che non ha mai incontrato in training.
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # center=False: i frame partono al campione 0, esattamente come sul Pico.
    # Con il default (center=True) librosa padda n_fft//2 campioni all'inizio
    # e tutti i frame risultano sfasati di mezza finestra.
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, center=False
    )

    # top_db=None: il default e' 80, che applica un clipping RELATIVO al massimo
    # del singolo spettrogramma, non replicabile sul Pico. Qui vogliamo solo il
    # clipping assoluto fatto sotto con np.clip.
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, top_db=None)
    log_mel = np.clip(log_mel, MIN_DB, MAX_DB)
    log_mel = (log_mel - MIN_DB) / (MAX_DB - MIN_DB)
    
    return log_mel[..., np.newaxis]

if __name__ == "__main__":
    print("Inizio estrazione feature ottimizzata per microcontrollori...\n")
    
    X, y = [], []
    file_processati = 0
    file_scartati = 0
    
    for label_idx, folder in enumerate(CATEGORIE):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path): continue
            
        file_audio_validi = [f for f in os.listdir(folder_path) if f.lower().endswith((".wav", ".mp3"))]
        
        for filename in tqdm(file_audio_validi, desc=f"Elaborando {folder: <16}", unit="file"):
            filepath = os.path.join(folder_path, filename)
            try:
                X.append(process_audio(filepath))
                y.append(label_idx)
                file_processati += 1
            except Exception as e:
                tqdm.write(f"   [!] Errore nel file {filename}: {e}")
                file_scartati += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    classes = np.array(CATEGORIE)

    np.savez(OUTPUT_FILE, X=X, y=y, classes=classes)
    
    print(f"\nMatrice X generata: {X.shape} (File, Mel-Bands, Time-Frames, Canale)")