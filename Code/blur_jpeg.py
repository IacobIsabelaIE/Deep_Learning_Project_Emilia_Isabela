import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import io

def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel with proper normalization."""
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    x, y = tf.meshgrid(x, x)
    
    # Calculate 2D gaussian
    kernel = tf.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # Normalize the kernel so it sums to 1
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Expand dimensions for depthwise conv2d
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    return kernel

def apply_gaussian_blur(image, probability=0.5):
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
        
        # Remove the batch dimension and ensure values stay in valid range
        blurred = tf.squeeze(blurred, 0)
        if tf.reduce_max(image) > 1.0:
            blurred = tf.clip_by_value(blurred, 0, 255)
        else:
            blurred = tf.clip_by_value(blurred, 0, 1)
        return blurred
    return image

def apply_jpeg_compression(image, probability=0.5):
    """Apply random JPEG compression with given probability."""
    def jpeg_compression_fn(image, quality):
        """Helper function to apply JPEG compression."""
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Encode and decode JPEG
        jpeg_encoded = tf.io.encode_jpeg(image, quality=int(quality))
        jpeg_decoded = tf.io.decode_jpeg(jpeg_encoded, channels=3)
        
        # Convert back to float32
        if image.max() <= 1.0:
            jpeg_decoded = jpeg_decoded.astype(np.float32) / 255.0
        else:
            jpeg_decoded = jpeg_decoded.astype(np.float32)
            
        return jpeg_decoded
    
    if tf.random.uniform([]) < probability:
        # Generate random quality and call jpeg_compression_fn with tf.py_function
        quality = tf.random.uniform([], minval=30, maxval=101, dtype=tf.int32)
        image = tf.py_function(func=jpeg_compression_fn, inp=[image, quality], Tout=tf.float32)
    return image


def apply_combined_effects(image, blur_prob=0.5, jpeg_prob=0.5):
    """Apply both blur and JPEG compression with independent probabilities."""
    # Apply blur first (if selected)
    image = apply_gaussian_blur(image, blur_prob)
    
    # Then apply JPEG compression (if selected)
    image = apply_jpeg_compression(image, jpeg_prob)
    
    return image

def load_data(base_dir, target_size=(224, 224), batch_size=32, num_classes=2, blur_prob=0.5, jpeg_prob=0.5):
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
    
    # Apply transformations to images
    def apply_effects_to_batch(images, labels):
        processed_images = tf.map_fn(
            lambda x: apply_combined_effects(x, blur_prob, jpeg_prob),
            images,
            dtype=tf.float32
        )
        return processed_images, labels

    # Map the transformations to the dataset
    test_dataset = base_dataset.map(apply_effects_to_batch)

    return test_dataset, total_test_samples