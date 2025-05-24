import tensorflow as tf

def get_tensorflow_datasets(train_dir, test_dir, img_size=224, batch_size=32):
    preprocess = tf.keras.applications.resnet50.preprocess_input  

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical'
    ).map(lambda x, y: (preprocess(x), y))

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical'
    ).map(lambda x, y: (preprocess(x), y))

    return train_ds, test_ds
