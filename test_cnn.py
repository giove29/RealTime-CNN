import os
import numpy as np
import librosa
import tensorflow as tf

# --- 1. CONFIGURAZIONE ---
MODEL_PATH = "smartcity_model.h5"
AUDIO_PATH = "test.wav"  # Puoi cambiare in "test.mp3" a piacimento

# Gli stessi identici parametri dell'addestramento
SAMPLE_RATE = 16000
DURATION = 2.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

N_MELS = 40
HOP_LENGTH = 512
N_FFT = 1024

MIN_DB = -80.0
MAX_DB = 20.0

# Mappatura esatta usata nel training
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
    """Prepara il file audio per l'inferenza mantenendo i criteri di training."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file '{file_path}' non esiste.")

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Adatta la lunghezza a 2.0 secondi esatti
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    elif len(audio) < SAMPLES_PER_TRACK:
        audio = np.pad(audio, (0, SAMPLES_PER_TRACK - len(audio)), 'constant')
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    
    # Normalizzazione globale fissa
    log_mel = np.clip(log_mel, MIN_DB, MAX_DB)
    log_mel = (log_mel - MIN_DB) / (MAX_DB - MIN_DB)
    
    # Aggiunge il canale (shape finale: 1, 40, 63, 1) per il batch della CNN
    return log_mel[np.newaxis, ..., np.newaxis]

if __name__ == "__main__":
    print(f"Caricamento del modello '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {e}")
        exit(1)

    print(f"Elaborazione del file audio '{AUDIO_PATH}'...")
    try:
        input_data = process_audio(AUDIO_PATH)
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione audio: {e}")
        exit(1)

    # --- INFERENZA ---
    print("Esecuzione classificazione...")
    predictions = model.predict(input_data, verbose=0)[0]
    
    # Trova l'indice con la probabilità più alta
    predicted_idx = np.argmax(predictions)
    predicted_class = CATEGORIE[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    # --- STAMPA RISULTATI ---
    print("\n" + "="*40)
    print("🎯 RISULTATO ANALISI")
    print("="*40)
    print(f"CLASSIFICAZIONE:  {predicted_class}")
    print(f"CONFIDENZA:       {confidence:.2f}%\n")
    
    print("Dettaglio probabilità per tutte le classi:")
    print("-" * 40)
    
    # Ordina le classi dalla più probabile alla meno probabile
    class_probs = list(zip(CATEGORIE, predictions))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    
    for category, prob in class_probs:
        print(f"{category:.<25} {prob*100:>6.2f}%")
    print("="*40)