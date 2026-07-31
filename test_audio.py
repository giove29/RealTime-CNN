import numpy as np
import librosa
import tensorflow as tf
import os

# Nascondi i log di TensorFlow (opzionale)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Caricamento del Modello e del Vocabolario
print("Caricamento del modello e delle classi...")
model = tf.keras.models.load_model("audio_classifier.keras")
classes = np.load("classes.npy")

# Parametri identici a quelli dell'addestramento
SR = 16000
CHUNK_SAMPLES = SR * 1  # 1 secondo
N_MFCC = 32
HOP_LENGTH = 512

def process_and_predict(file_path):
    print(f"\nAnalisi del file: {file_path}")
    
    # 2. Caricamento file audio (librosa legge automaticamente sia WAV che MP3)
    audio, _ = librosa.load(file_path, sr=SR)
    
    # Se l'audio dura meno di 1 secondo, lo riempiamo di zeri (silenzio) per arrivare a 1s
    if len(audio) < CHUNK_SAMPLES:
        audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        
    # 3. Taglio in frammenti da 1 secondo
    chunks = []
    for i in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunk = audio[i : i + CHUNK_SAMPLES]
        
        # Estrazione feature MFCC
        mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        chunks.append(mfcc)

    X_test = np.array(chunks)
    
    # Aggiunta dimensione canale colore (N, 16, 32, 1)
    X_test = X_test[..., np.newaxis]
    
# 4. NORMALIZZAZIONE INDIPENDENTE
    for j in range(len(X_test)):
        mean_val = np.mean(X_test[j])
        std_val = np.std(X_test[j]) + 1e-8
        X_test[j] = (X_test[j] - mean_val) / std_val
    
    # 5. Previsione
    predictions = model.predict(X_test, verbose=0)
    
    # Stampiamo i risultati intermedi per ogni secondo
    print("\n--- Analisi secondo per secondo ---")
    for i, pred in enumerate(predictions):
        best_idx = np.argmax(pred)
        confidence = pred[best_idx] * 100
        print(f"Secondo {i+1}: {classes[best_idx]} ({confidence:.1f}%)")
        
    # 6. Il verdetto finale (Media delle probabilità)
    avg_predictions = np.mean(predictions, axis=0)
    final_idx = np.argmax(avg_predictions)
    final_confidence = avg_predictions[final_idx] * 100
    
    print("\n====================================")
    print(f"CLASSIFICAZIONE FINALE: {classes[final_idx].upper()}")
    print(f"Affidabilità media: {final_confidence:.1f}%")
    print("====================================\n")


# Logica di selezione del file (priorità al .wav)
wav_file = "test.wav"
mp3_file = "test.mp3"

if os.path.exists(wav_file):
    print(f"Trovato {wav_file}, avvio test...")
    process_and_predict(wav_file)
elif os.path.exists(mp3_file):
    print(f"Trovato {mp3_file}, avvio test...")
    process_and_predict(mp3_file)
else:
    print(f"Errore: Non ho trovato né '{wav_file}' né '{mp3_file}' nella cartella.")