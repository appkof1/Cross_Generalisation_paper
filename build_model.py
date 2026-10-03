import tensorflow as tf

def build_model():
    """
    Builds an instance of the breast classification model with VGG16
    architecture pre-trained with ImageNet data as a base.

    Returns:
        tf.Keras.Model: The breast cancer classification model. 
    """
    base_model = tf.keras.applications.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_shape=(224, 224, 3))
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    return model
