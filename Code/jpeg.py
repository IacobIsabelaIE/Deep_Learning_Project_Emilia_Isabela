import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import io

import tensorflow as tf

def apply_jpeg_compression(image, probability=0.5):
    """Apply random JPEG compression with given probability."""
    if tf.random.uniform([]) < probability:
        # Generate random quality from uniform distribution [30, 100]
        quality = tf.random.uniform([], minval=30, maxval=101, dtype=tf.int32)

        # Ensure the image is in uint8 format (required for JPEG compression)
        if tf.reduce_max(image) <= 1.0:
            image = image * 255.0  # Convert from [0, 1] range to [0, 255] range
        # image = tf.cast(image, tf.uint8)

        # Apply JPEG compression using tf.image.adjust_jpeg_quality
        compressed_image = tf.image.adjust_jpeg_quality(image, quality)

        # Optionally, normalize back to [0, 1]
        # compressed_image = tf.cast(compressed_image, tf.float32) / 255.0

        return compressed_image

    return image


def load_data(base_dir, target_size=(224, 224), batch_size=32, num_classes=2, probability=0.5):
    datagen = ImageDataGenerator()

    test_generators = []
    test_sizes = []

    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            test_dir = os.path.join(subfolder_path, 'test')

            if os.path.exists(test_dir):
                test_gen = datagen.flow_from_directory(
                    test_dir,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode="categorical"
                )
                test_generators.append(test_gen)
                test_sizes.append(test_gen.samples)

    total_test_samples = sum(test_sizes)

    def combine_generators(generators):
        for gen in generators:
            yield from gen
    
    # Create base dataset from generators
    base_dataset = tf.data.Dataset.from_generator(
        lambda: combine_generators(test_generators),
        output_signature=(
            tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),  
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)       
        )
    )
    
    # Apply random JPEG compression to images
    def apply_compression_to_batch(images, labels):
        processed_images = tf.map_fn(
            lambda x: apply_jpeg_compression(x, probability=probability),
            images,
            dtype=tf.float32
        )
        return processed_images, labels

    # Map the compression transformation to the dataset
    test_dataset = base_dataset.map(apply_compression_to_batch)

    return test_dataset, total_test_samples