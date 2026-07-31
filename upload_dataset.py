import os
import shutil
import pandas as pd

# Percorsi di riferimento
CSV_PATH = "meta/esc50.csv"
AUDIO_DIR = "audio/"
DEST_DIR = "dataset"

# Categorie selezionate mappate sui nomi esatti del dataset ESC-50
allowed_categories = {
    "dog", "cat", "rain", "crackling_fire", "crickets", 
    "chirping_birds", "wind", "thunderstorm", "crying_baby", 
    "clapping", "footsteps", "can_opening", "glass_breaking", "helicopter", 
    "siren", "car_horn", "engine", "train", "church_bells", 
    "airplane", "fireworks"
}

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File non trovato: {CSV_PATH}. Verifica il percorso della repository ESC-50.")

# Caricamento e filtraggio dei metadati
df = pd.read_csv(CSV_PATH)
filtered_df = df[df['category'].isin(allowed_categories)]

copied_count = 0
for _, row in filtered_df.iterrows():
    filename = row['filename']
    category = row['category']
    
    class_folder = os.path.join(DEST_DIR, category)
    os.makedirs(class_folder, exist_ok=True)
    
    src_path = os.path.join(AUDIO_DIR, filename)
    dst_path = os.path.join(class_folder, filename)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied_count += 1

print(f"Filtraggio completato: copiati {copied_count} file audio distribuiti su {len(filtered_df['category'].unique())} categorie all'interno di '{DEST_DIR}'.")