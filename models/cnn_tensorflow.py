import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def get_pretrained_model_tf(input_shape=(224, 224, 3), num_classes=4):
    # Charger ResNet50 sans la couche de classification finale
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Geler toutes les couches initialement
    base_model.trainable = False

    # Dégeler les dernières couches pour fine-tuning (par exemple, block5)
    for layer in base_model.layers[-10:]:  # Dégeler les 10 dernières couches
        layer.trainable = True

    # Construire le modèle avec régularisation
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Ajout de L2
        layers.Dropout(0.5),  # Conserver Dropout pour réduire l'overfitting
        layers.Dense(num_classes, activation='softmax')  # 4 classes pour tumeurs cérébrales
    ])

    return model