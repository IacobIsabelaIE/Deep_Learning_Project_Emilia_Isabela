import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    x, y = tf.meshgrid(x, x)
    
    # Calculate 2D gaussian
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = tf.exp(-(x**2 + y**2) / (2.0 * sigma**2)) * normal
    
    # Normalize the kernel
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Expand dimensions for tf.nn.conv2d
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    return tf.tile(kernel, [1, 1, 3, 1])

def apply_random_blur(image, probability=0.5):
    """Apply random Gaussian blur with given probability."""
    if tf.random.uniform([]) < probability:
        # Generate random sigma from uniform distribution [0, 3]
        sigma = tf.random.uniform([], 0, 3)
        
        # Calculate kernel size based on sigma (ensure odd number)
        kernel_size = tf.cast(tf.math.ceil(sigma * 3) * 2 + 1, tf.int32)
        kernel_size = tf.maximum(kernel_size, 3)
        
        # Create Gaussian kernel
        kernel = gaussian_kernel(kernel_size, sigma)
        
        # Reshape image for convolution
        image = tf.expand_dims(image, 0)
        
        # Apply convolution separately for each channel
        blurred = tf.nn.depthwise_conv2d(
            image,
            kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        
        return tf.squeeze(blurred, 0)
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
    
    # Apply random blur transformation to images
    def apply_blur_to_batch(images, labels):
        processed_images = tf.map_fn(
            lambda x: apply_random_blur(x, probability=probability),
            images,
            dtype=tf.float32
        )
        return processed_images, labels

    # Map the blur transformation to the dataset
    test_dataset = base_dataset.map(apply_blur_to_batch)

    return test_dataset, total_test_samples