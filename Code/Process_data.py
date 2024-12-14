def load_data(base_dir, target_size=(224, 224), batch_size=32, num_classes=2):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import tensorflow as tf

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generators = []
    val_generators = []
    train_sizes = []
    val_sizes = []

    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            train_dir = os.path.join(subfolder_path, 'train')
            val_dir = os.path.join(subfolder_path, 'val')

            if os.path.exists(train_dir):
                train_gen = datagen.flow_from_directory(
                    train_dir,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode="categorical"
                )
                train_generators.append(train_gen)
                train_sizes.append(train_gen.samples)

            if os.path.exists(val_dir):
                val_gen = datagen.flow_from_directory(
                    val_dir,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode="categorical"
                )
                val_generators.append(val_gen)
                val_sizes.append(val_gen.samples)

    total_train_samples = sum(train_sizes)
    total_val_samples = sum(val_sizes)

    def combine_generators(generators):
        for gen in generators:
            yield from gen

    train_dataset = tf.data.Dataset.from_generator(
        lambda: combine_generators(train_generators),
        output_signature=(
            tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)       
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: combine_generators(val_generators),
        output_signature=(
            tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),  
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)       
        )
    )

    return train_dataset, val_dataset, total_train_samples, total_val_samples
