import os
import numpy as np
import tensorflow as tf

MODEL_INPUT = "smartcity_model.h5"
CALIBRATION_INPUT = "calibration_data.npy"
TFLITE_OUTPUT = "smartcity_sensor_int8.tflite"
CPP_OUTPUT = "model_data.cc"

def representative_dataset():
    calibration_data = np.load(CALIBRATION_INPUT)
    for i in range(len(calibration_data)):
        data_sample = np.expand_dims(calibration_data[i], axis=0).astype(np.float32)
        yield [data_sample]

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_INPUT)

    # La batch dimension a None costringe il convertitore a calcolare la forma
    # del reshape a runtime, generando SHAPE + STRIDED_SLICE + PACK: tre
    # operatori dinamici che TFLite Micro non gestisce. Va fissata a 1.
    feature_shape = tuple(model.inputs[0].shape[1:])
    print(f"Conversione con forma di input fissa: {(1,) + feature_shape}")

    converter = None

    # Strada 1: ricostruire il modello con batch fissa e ricopiare i pesi.
    # E' la piu' pulita perche' resta sul percorso from_keras_model.
    try:
        fixed = tf.keras.models.clone_model(
            model,
            input_tensors=tf.keras.Input(batch_shape=(1,) + feature_shape)
        )
        fixed.set_weights(model.get_weights())
        converter = tf.lite.TFLiteConverter.from_keras_model(fixed)
        print("  -> batch fissata con clone_model")
    except Exception as e:
        print(f"  clone_model non utilizzabile ({e}), passo al congelamento")

    # Strada 2: concrete function con i pesi CONGELATI in costanti.
    # Senza il congelamento restano READ_VARIABLE e il calibratore fallisce.
    if converter is None:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2)
        run = tf.function(lambda x: model(x, training=False))
        concrete = run.get_concrete_function(
            tf.TensorSpec((1,) + feature_shape, tf.float32))
        frozen = convert_variables_to_constants_v2(concrete)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        print("  -> batch fissata con concrete function congelata")
    
    # Quantizzazione Full-Integer 8-bit
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(TFLITE_OUTPUT, 'wb') as f:
        f.write(tflite_model)

    # Verifica degli operatori: da fare SEMPRE prima di toccare il Pico.
    # Se qui compare qualcosa che non e' nel resolver di main.cpp, sul
    # microcontrollore AllocateTensors fallisce o l'inferenza si blocca.
    ATTESI = {"CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D", "AVERAGE_POOL_2D",
              "RESHAPE", "FULLY_CONNECTED", "SOFTMAX", "QUANTIZE", "DEQUANTIZE"}
    # DELEGATE non e' un operatore del file: e' XNNPACK che l'interprete del PC
    # applica a runtime per accelerare. Sul Pico non esiste, va ignorato.
    IGNORA = {"DELEGATE"}
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    print("\nOperatori nel modello:")
    problemi = []
    for d in interp._get_ops_details():
        nome = d['op_name']
        if nome in IGNORA:
            continue
        ok = nome in ATTESI
        print(f"  {'OK ' if ok else '>>>'} {nome}")
        if not ok:
            problemi.append(nome)
    if problemi:
        print(f"\nATTENZIONE: operatori non previsti -> {sorted(set(problemi))}")
        print("Non flashare: correggi il modello prima.")
    else:
        print("\nTutti gli operatori sono supportati dal resolver di main.cpp.")
    
    dim_kb = len(tflite_model) / 1024
    print(f"Modello compresso a {dim_kb:.2f} KB")

    # Generazione file C++
    hex_array = [f"0x{b:02x}" for b in tflite_model]
    lines = [", ".join(hex_array[i:i+12]) for i in range(0, len(hex_array), 12)]
    
    c_code = (
        "// Modello CNN Nano Quantizzato a 8-bit per Raspberry Pi Pico\n"
        "#include <cstdint>\n\n"
        # 'extern' obbligatorio: in C++ un const a scope di namespace ha
        # linkage INTERNO, quindi senza extern il linker non lo trova da main.cpp
        "alignas(8) extern const unsigned char g_model[] = {\n  " + 
        ",\n  ".join(lines) + 
        "\n};\n\n"
        f"extern const unsigned int g_model_len = {len(tflite_model)};\n"
    )

    with open(CPP_OUTPUT, 'w') as f:
        f.write(c_code)

    print(f"File '{CPP_OUTPUT}' generato. Copialo nel tuo progetto Pico!")