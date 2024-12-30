import os
import random
import shutil

def split_val_to_test(base_path, main_dirs, split_ratio=0.5):
    """
    Move a portion of images from 'val' folders to 'test' folders.

    Parameters:
        base_path (str): The root directory containing the main directories.
        main_dirs (list): List of main directories to process.
        split_ratio (float): Fraction of images to move from 'val' to 'test'. Default is 0.5 (half).
    """
    for main_dir in main_dirs:
        ai_test_folder = os.path.join(base_path, main_dir, 'test', 'ai')
        nature_test_folder = os.path.join(base_path, main_dir, 'test', 'nature')

        os.makedirs(ai_test_folder, exist_ok=True)
        os.makedirs(nature_test_folder, exist_ok=True)

        for category in ['ai', 'nature']:
            val_folder = os.path.join(base_path, main_dir, 'val', category)
            test_folder = os.path.join(base_path, main_dir, 'test', category)

            if not os.path.exists(val_folder):
                print(f"Warning: {val_folder} does not exist. Skipping...")
                continue

            val_images = [img for img in os.listdir(val_folder) if os.path.isfile(os.path.join(val_folder, img))]
            random.shuffle(val_images)

            num_images_to_move = int(len(val_images) * split_ratio)
            images_to_move = val_images[:num_images_to_move]

            for image in images_to_move:
                src_path = os.path.join(val_folder, image)
                dst_path = os.path.join(test_folder, image)
                shutil.move(src_path, dst_path)

            print(f"Moved {num_images_to_move} images from {val_folder} to {test_folder}")
