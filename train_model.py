import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURAZIONE GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU RILEVATA: Addestramento accelerato attivo su {len(gpus)} GPU.")
    except RuntimeError as e:
        print(f"Errore configurazione GPU: {e}")
else:
    print("⚠️ NESSUNA GPU RILEVATA: L'addestramento avverrà sulla CPU.")

# --- 2. CONFIGURAZIONE ---
FEATURES_FILE = "dataset_features.npz"
MODEL_OUTPUT = "smartcity_model.h5"
CALIBRATION_OUTPUT = "calibration_data.npy"
EPOCHS = 30
BATCH_SIZE = 16  # Batch ridotto per raddoppiare l'aggiornamento dei pesi

if __name__ == "__main__":
    print(f"\nCaricamento dati da {FEATURES_FILE}...")
    data = np.load(FEATURES_FILE)
    X = data['X']
    y = data['y']
    classes = data['classes']
    NUM_CLASSES = len(classes)
    
    # Split rigoroso dei dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    input_shape = X_train[0].shape
    
    # Salviamo 100 sample per la calibrazione della quantizzazione
    np.save(CALIBRATION_OUTPUT, X_train[:100])
    print(f"Dati caricati. Shape input: {input_shape} (2 secondi di audio).")

    # --- 3. DATA AUGMENTATION ---
    # Genera micro-variazioni a ogni epoca per distruggere l'overfitting
    datagen = ImageDataGenerator(
        width_shift_range=0.1,   # Traslazione temporale (simula suoni in ritardo/anticipo)
        height_shift_range=0.05, # Micro-variazione di pitch
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # --- 4. ARCHITETTURA MODELLO SOTA PER MICROCONTROLLORI ---
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        
        # Dropout e regolarizzazione L2 per forzare la generalizzazione
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.01))
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # --- 5. LEARNING RATE SCHEDULER ---
    # Affina la mira se la rete va in stallo
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=4, 
        verbose=1, 
        min_lr=1e-5
    )

    # --- 6. ADDESTRAMENTO ---
    print(f"\nInizio addestramento per {EPOCHS} epoche...")
    
    # Uso di datagen.flow per iniettare i dati "aumentati" in tempo reale
    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[lr_scheduler]
    )

    model.save(MODEL_OUTPUT)
    print(f"\nAddestramento terminato! Modello salvato come '{MODEL_OUTPUT}'.")