import os
import numpy as np
import tensorflow as tf

# --- CONFIGURAZIONE ---
MODEL_INPUT = "smartcity_model.h5"
CALIBRATION_INPUT = "calibration_data.npy"
TFLITE_OUTPUT = "smartcity_sensor_int8.tflite"
CPP_OUTPUT = "model_data.cc"

def representative_dataset():
    """
    Generatore essenziale per la Full Integer Quantization.
    Fornisce al convertitore i 100 sample reali che abbiamo salvato 
    nella fase di training per calcolare l'escursione minima e massima 
    (range dinamico) delle attivazioni e scalare i pesi senza perdere precisione.
    """
    calibration_data = np.load(CALIBRATION_INPUT)
    for i in range(len(calibration_data)):
        # TFLite richiede esplicitamente che il formato di input sia float32
        data_sample = np.expand_dims(calibration_data[i], axis=0).astype(np.float32)
        yield [data_sample]

if __name__ == "__main__":
    if not os.path.exists(MODEL_INPUT) or not os.path.exists(CALIBRATION_INPUT):
        print("❌ Errore: Assicurati di aver prima completato l'addestramento (File 2)!")
        exit(1)

    print(f"Caricamento modello {MODEL_INPUT}...")
    model = tf.keras.models.load_model(MODEL_INPUT)
    
    print("Inizio conversione e quantizzazione Full INT8 in corso...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 1. Ottimizzazione di default per ridurre le dimensioni
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. Assegnazione del dataset rappresentativo
    converter.representative_dataset = representative_dataset
    
    # 3. Forzatura stretta: SE un'operazione non è convertibile in INT8, fermati con errore
    # Questo assicura che il modello sia 100% compatibile con i microcontrollori senza FPU
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 4. Forzatura I/O: Anche l'input (lo spettrogramma) e l'output (le probabilità) devono essere INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Esecuzione della conversione matematica
    tflite_model = converter.convert()
    
    # Salvataggio fisico del file TFLite (Utile se vuoi testarlo su PC prima del deploy)
    with open(TFLITE_OUTPUT, 'wb') as f:
        f.write(tflite_model)
    
    dim_kb = len(tflite_model) / 1024
    print(f"✅ Modello TFLite salvato con successo: {TFLITE_OUTPUT}")
    print(f"📏 Dimensione finale compressa: {dim_kb:.2f} KB (Perfetto per la RAM del Pico!)")

    # --- GENERAZIONE DEL CODICE SORGENTE C/C++ ---
    print("\nGenerazione del file sorgente C++ (Hex Array)...")
    
    # Leggiamo il modello binario appena creato
    with open(TFLITE_OUTPUT, 'rb') as f:
        tflite_content = f.read()

    # Formattiamo l'array esadecimale riga per riga per renderlo leggibile
    hex_array = [f"0x{b:02x}" for b in tflite_content]
    lines = [", ".join(hex_array[i:i+12]) for i in range(0, len(hex_array), 12)]
    
    c_code = (
        "// Modello CNN Quantizzato a 8-bit per Raspberry Pi Pico\n"
        "// Autogenerato tramite TensorFlow Lite Micro\n\n"
        "#include <cstdint>\n\n"
        "alignas(8) const unsigned char g_model[] = {\n  " + 
        ",\n  ".join(lines) + 
        "\n};\n\n"
        f"const unsigned int g_model_len = {len(tflite_content)};\n"
    )

    # Salvataggio del file .cc
    with open(CPP_OUTPUT, 'w') as f:
        f.write(c_code)

    print(f"✅ Operazione conclusa! Il file '{CPP_OUTPUT}' è stato generato.")
    print("Ora puoi spostare 'model_data.cc' direttamente nel tuo progetto C++ del Raspberry Pi Pico!")