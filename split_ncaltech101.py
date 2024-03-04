import os
import shutil
import random

def split_dataset(dataset_dir, split_ratio=(0.7, 0.15, 0.15), random_seed=None):
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print("Dataset directory not found.")
        return

    # Create directories for train, validation, and test sets
    train_dir = os.path.join(dataset_dir, 'training')
    validation_dir = os.path.join(dataset_dir, 'validation')
    test_dir = os.path.join(dataset_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of class labels
    class_labels = os.listdir(os.path.join(dataset_dir, 'Caltech101'))

    # Split each class into train, validation, and test sets
    for label in class_labels:
        label_dir = os.path.join(dataset_dir, 'Caltech101', label)
        image_files = os.listdir(label_dir)

        # Shuffle the image files randomly
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(image_files)

        # Calculate the number of images for each split
        num_images = len(image_files)
        num_train = int(num_images * split_ratio[0])
        num_validation = int(num_images * split_ratio[1])
        num_test = num_images - num_train - num_validation

        # Assign images to train, validation, and test sets
        train_images = image_files[:num_train]
        validation_images = image_files[num_train:num_train + num_validation]
        test_images = image_files[num_train + num_validation:]

        # Copy images to the respective directories
        for image_file in train_images:
            src = os.path.join(label_dir, image_file)
            dst = os.path.join(train_dir, label)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)

        for image_file in validation_images:
            src = os.path.join(label_dir, image_file)
            dst = os.path.join(validation_dir, label)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)

        for image_file in test_images:
            src = os.path.join(label_dir, image_file)
            dst = os.path.join(test_dir, label)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)

    # Copy annotation files to respective directories
    annotations_src = os.path.join(dataset_dir, 'Caltech101_annotations')
    annotations_dst_train = os.path.join(train_dir, 'annotations')
    annotations_dst_validation = os.path.join(validation_dir, 'annotations')
    annotations_dst_test = os.path.join(test_dir, 'annotations')
    shutil.copytree(annotations_src, annotations_dst_train)
    shutil.copytree(annotations_src, annotations_dst_validation)
    shutil.copytree(annotations_src, annotations_dst_test)

    print("Dataset split completed.")

# Example usage:
dataset_dir = "/DATA/mccha/DVS/ncaltech101/"
split_dataset(dataset_dir, split_ratio=(0.8, 0.1, 0.1), random_seed=42)
