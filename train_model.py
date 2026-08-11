import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_FILE = "dataset_features.npz"
MODEL_OUTPUT = "smartcity_model.h5"
CALIBRATION_OUTPUT = "calibration_data.npy"

EPOCHS = 150          # tanto interviene EarlyStopping: meglio abbondare
BATCH_SIZE = 32       # 16 rendeva rumorose le statistiche di BatchNormalization
WIDTH = 2             # moltiplicatore dei filtri: prova 2 se resta sotto-adattato
L2 = 1e-4             # era 1e-2, troppo per un modello che non sovra-adatta


def build_model(input_shape, num_classes, width=1):
    """Sole convoluzioni standard: niente DepthwiseConv2D (si blocca nei kernel
    int8 di TFLite Micro) e niente GlobalAveragePooling2D (diventa MEAN).
    BatchNormalization PRIMA dell'attivazione, cosi' il convertitore la fonde
    nella convoluzione invece di lasciarla come coppia MUL + ADD."""
    f1, f2, f3 = 8 * width, 16 * width, 32 * width
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(f1, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f2, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(f3, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.AveragePooling2D(pool_size=(h, w))(x)
    x = layers.Reshape((c,))(x)           # forma esplicita: Flatten genera
    x = layers.Dropout(0.3)(x)            # SHAPE/STRIDED_SLICE/PACK
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=regularizers.l2(L2))(x)
    return models.Model(inputs, outputs)


if __name__ == "__main__":
    data = np.load(FEATURES_FILE)
    X, y, classes = data['X'], data['y'], data['classes']
    NUM_CLASSES = len(classes)

    print("Distribuzione delle classi:")
    for i, c in enumerate(classes):
        print(f"  {str(c):<20} {int((y == i).sum()):5d}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    input_shape = X_train[0].shape
    print(f"\nForma dell'input: {input_shape}   train {len(X_train)}  test {len(X_test)}")
    np.save(CALIBRATION_OUTPUT, X_train[:100])

    # Compensa lo sbilanciamento: senza, la rete impara a non scommettere mai
    # sulle classi rare perche' le conviene per l'accuratezza globale.
    pesi = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(pesi))
    print("Pesi per classe:", {int(k): round(float(v), 2) for k, v in class_weight.items()})

    # Solo spostamento temporale. Lo spostamento in frequenza altera l'identita'
    # del suono: su uno spettrogramma mel equivale a trasporre l'altezza.
    datagen = ImageDataGenerator(width_shift_range=0.1, fill_mode='nearest')
    datagen.fit(X_train)

    model = build_model(input_shape, NUM_CLASSES, WIDTH)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    cb = [
        # Sorveglia val_loss, non val_accuracy: l'accuratezza su un insieme di
        # validazione piccolo e' rumorosa e faceva scendere il learning rate
        # reagendo al rumore invece che a un vero plateau.
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                                    verbose=1, min_lr=1e-6),
        # restore_best_weights: senza, si salva l'ultima epoca invece della
        # migliore. Nel training precedente costava mezzo punto di accuratezza.
        callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1,
                                restore_best_weights=True),
    ]

    hist = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=cb,
        verbose=2,
    )

    model.save(MODEL_OUTPUT)
    print(f"\nModello salvato come '{MODEL_OUTPUT}'.")
    print(f"Migliore val_accuracy: {max(hist.history['val_accuracy']):.4f}")

    # L'accuratezza globale con classi sbilanciate mente. Quello che conta e'
    # il richiamo per classe e chi si confonde con chi.
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    nomi = [str(c) for c in classes]
    print("\n" + classification_report(y_test, y_pred, target_names=nomi, digits=3))
    print("Matrice di confusione (righe = vere, colonne = predette):")
    cm = confusion_matrix(y_test, y_pred)
    print("            " + " ".join(f"{n[:6]:>7}" for n in nomi))
    for i, n in enumerate(nomi):
        print(f"{n[:11]:<11} " + " ".join(f"{v:7d}" for v in cm[i]))