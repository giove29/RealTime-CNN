import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("audio_classifier.keras")
print("Modello CNN caricato correttamente.")

# Carichiamo i dati reali salvati dallo script di addestramento
X_train = np.load("X_train.npy")

def representative_dataset_gen():
    # Passiamo alla rete 100 campioni REALI per farle capire i valori minimi e massimi
    for i in range(100):
        # Prendiamo un singolo campione e aggiungiamo la dimensione del batch: (1, 16, 32, 1)
        sample = X_train[i : i+1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Quantizzazione INT8 completa
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Conversione e quantizzazione in corso...")
tflite_quant_model = converter.convert()

output_path = "audio_model_quantized.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_quant_model)

file_size_kb = os.path.getsize(output_path) / 1024
print(f"\nModello quantizzato salvato con successo come '{output_path}'!")
print(f"Dimensione finale del file: {file_size_kb:.2f} KB (Pronto per il Pico!)")