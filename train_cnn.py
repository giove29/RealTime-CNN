import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

print("Caricamento dei tensori in corso...")
X = np.load("X_data.npy")
y = np.load("y_labels.npy")
classes = np.load("classes.npy")
num_classes = len(classes)


# 1. Normalizzazione Indipendente per ogni frammento (Perfetto per il Pico)
# Al posto di X_mean = np.mean(X)... usa questo:
print("Normalizzazione dei singoli frammenti...")
for i in range(len(X)):
    mean_val = np.mean(X[i])
    std_val = np.std(X[i]) + 1e-8 # 1e-8 evita la divisione per zero nel silenzio assoluto
    X[i] = (X[i] - mean_val) / std_val

# 2. Divisione in set di addestramento (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save("X_train.npy", X_train)

# 3. Rete potenziata ma ottimizzata per Microcontrollore
input_shape = X.shape[1:] 

model = models.Sequential([
    layers.Input(shape=input_shape),
    
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    # Dropout aumentato a 0.55 per forzare una maggiore generalizzazione
    layers.Dropout(0.55), 
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. LA MAGIA: Impostazione del Checkpoint
# Monitoriamo 'val_accuracy' e salviamo sovrascrivendo il file SOLO se il valore migliora
checkpoint = ModelCheckpoint(
    filepath="audio_classifier.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

print("\nAddestramento della CNN in corso...")
# Passiamo il checkpoint alla funzione fit tramite l'argomento 'callbacks'
history = model.fit(
    X_train, y_train, 
    epochs=40, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# 5. Valutazione finale (Carichiamo i pesi del modello migliore appena salvato)
best_model = tf.keras.models.load_model("audio_classifier.keras")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

print(f"\n--- RISULTATO ADDESTRAMENTO ---")
print(f"Precisione del MODELLO MIGLIORE sul set di test: {test_acc * 100:.2f}%")
print("Il file 'audio_classifier.keras' contiene la rete della tua epoca migliore.")