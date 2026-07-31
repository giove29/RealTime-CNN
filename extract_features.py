import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configurazioni
DATASET_PATH = "dataset"
SR = 16000             # Frequenza di campionamento (16 kHz, ideale per il Pico)
DURATION = 1           # Finestre da 1 secondo
CHUNK_SAMPLES = SR * DURATION 
N_MFCC = 32            # Numero di bande di frequenza (Asse Y dell'"immagine")
HOP_LENGTH = 512       # Risoluzione temporale (Asse X dell'"immagine")

X = []
y_text = []

print("Inizio estrazione feature audio... (potrebbe richiedere qualche minuto)")

# Dizionario di mapping: "nome_cartella" -> "macro_classe"
class_mapping = {
    # 0_Normale (Rumori di fondo, natura, meteo e normale attività urbana)
    "airplane": "0_Normale", 
    "can_opening": "0_Normale", 
    "car_passing": "0_Normale",
    "cat": "0_Normale", 
    "chirping_birds": "0_Normale", 
    "church_bells": "0_Normale",
    "clapping": "0_Normale", 
    "crickets": "0_Normale", 
    "crowd": "0_Normale",
    "footsteps": "0_Normale", 
    "rain": "0_Normale", 
    "thunderstorm": "0_Normale",
    "train": "0_Normale", 
    "wind": "0_Normale",

    # 1_Disturbo (Inquinamento acustico, eventi fastidiosi da loggare)
    "car_horn": "1_Disturbo", 
    "dog": "1_Disturbo",
    "engine": "1_Disturbo", 
    "helicopter": "1_Disturbo",

    # 2_Emergenza (Situazioni anomale che richiedono un controllo visivo)
    "crying_baby": "2_Emergenza", 
    "siren": "2_Emergenza",

    # 3_Pericolo_Critico (Allarmi rossi di sicurezza pubblica)
    "crackling_fire": "3_Pericolo_Critico",  # Fuoco non autorizzato in piazza
    "fireworks": "3_Pericolo_Critico",       # Spesso illegali o confondibili con spari
    "glass_breaking": "3_Pericolo_Critico"   # Vandalismo, effrazione o incidenti
}

print("Elaborazione e raggruppamento in Macro-Classi...")

for label in os.listdir(DATASET_PATH):
    # Se la cartella non è nel nostro mapping (es. suoni scartati come 'sneezing'), la saltiamo
    if label not in class_mapping:
        continue
        
    macro_class = class_mapping[label]
    folder_path = os.path.join(DATASET_PATH, label)
    
    for file in os.listdir(folder_path):
        if not file.lower().endswith((".wav", ".mp3")):
            continue
            
        file_path = os.path.join(folder_path, file)
        
        try:
            audio, _ = librosa.load(file_path, sr=SR)
            for i in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
                chunk = audio[i : i + CHUNK_SAMPLES]
                mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
                
                X.append(mfcc)
                # Invece di salvare il nome della cartella, salviamo la macro-classe!
                y_text.append(macro_class)
        except Exception as e:
            pass

# Conversione in array NumPy
X = np.array(X)
y_text = np.array(y_text)

# 1. Aggiungiamo una dimensione finale "fittizia" (il canale colore).
# Le CNN si aspettano una forma (altezza, larghezza, canali). Per noi è 1 canale (monocromatico).
# La forma finale diventerà circa (N_campioni, 16, 32, 1)
X = X[..., np.newaxis]

# 2. Codifica delle etichette testuali (es. "dog" -> 0, "rain" -> 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)

# Salvataggio dei tensori su disco per l'addestramento
np.save("X_data.npy", X)
np.save("y_labels.npy", y_encoded)
np.save("classes.npy", encoder.classes_)

print(f"\nOperazione completata con successo!")
print(f"Formato Tensore di Input (X): {X.shape}")
print(f"Categorie trovate ({len(encoder.classes_)}): {encoder.classes_}")
print("I file X_data.npy, y_labels.npy e classes.npy sono pronti per la CNN.")