# dataloader.py
import tensorflow as tf
import os
import glob

def create_label_from_filename(filename):
    """
    Creates label from filename:
    - Suffix '1': 1 (drowsy)
    - Suffix '0': 0 (notdrowsy)
    """
    # Remove file extension
    filename_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    # Get suffix number from filename
    if filename_without_ext.endswith('_1'):
        return 1
    elif filename_without_ext.endswith('_0'):
        return 0
    else:
        return 0  # Default

def get_data_pipelines(base_dir,
                       img_size=(224, 224),
                       batch_size=32,
                       seed=42,
                       shuffle_buffer=1000):
    """
    base_dir/
        train/ (mixed drowsy and notdrowsy files)
        val/   (mixed drowsy and notdrowsy files)
        test/  (mixed drowsy and notdrowsy files)
    
    Labels are created based on filename suffixes:
    - _1: 1 (drowsy)
    - _0: 0 (notdrowsy)
    """

    # 1) Get file paths manually
    def get_file_paths_and_labels(subdir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            return [], []
        
        file_paths = []
        labels = []
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            files = glob.glob(os.path.join(subdir_path, ext))
            for file_path in files:
                filename = os.path.basename(file_path)
                label = create_label_from_filename(filename)
                file_paths.append(file_path)
                labels.append(label)
        
        return file_paths, labels

    # 2) Create datasets
    train_files, train_labels = get_file_paths_and_labels('train')
    val_files, val_labels = get_file_paths_and_labels('val')
    test_files, test_labels = get_file_paths_and_labels('test')

    # 3) Create TensorFlow datasets
    def create_dataset(file_paths, labels, shuffle=True):
        if not file_paths:
            return None
        
        # Create dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        # Load and preprocess images
        def load_and_preprocess(file_path, label):
            # Load image
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, img_size)
            return image, label
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer, seed=seed)
        
        dataset = dataset.batch(batch_size)
        return dataset

    train_ds = create_dataset(train_files, train_labels, shuffle=True)
    val_ds = create_dataset(val_files, val_labels, shuffle=False)
    test_ds = create_dataset(test_files, test_labels, shuffle=False)

    # 4) Cache
    if train_ds:
        train_ds = train_ds.cache()
    if val_ds:
        val_ds = val_ds.cache()
    if test_ds:
        test_ds = test_ds.cache()

    # 5) Augment + normalize
    rotation_factor = 0.3      # 0.1 = %10, 0.2 = %20, 0.3 = %30
    zoom_factor = 0.3          # 0.1 = %10, 0.2 = %20, 0.3 = %30
    brightness_delta = 0.3     # 0.1 = low, 0.2 = mid, 0.3 = strong
    contrast_factor = 0.3      # 0.1 = low, 0.2 = mid, 0.3 = strong
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(rotation_factor),
        tf.keras.layers.RandomZoom(zoom_factor),
        #tf.keras.layers.RandomBrightness(brightness_delta),
        #tf.keras.layers.RandomContrast(contrast_factor),
    ])
    normalization = tf.keras.layers.Rescaling(1./255)

    # 6) Apply
    if train_ds:
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    if val_ds:
        val_ds = val_ds.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    if test_ds:
        test_ds = test_ds.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 7) Prefetch
    if train_ds:
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    if val_ds:
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    if test_ds:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    base_dir = "data"
    train_ds, val_ds, test_ds = get_data_pipelines(base_dir)

    # Test
    if train_ds:
        for imgs, labels in train_ds.take(1):
            print("Images:", imgs.shape, "Labels:", labels.shape)
            print("Labels:", labels.numpy())
            # Example: (32, 224, 224, 3) (32,)
            # Label format: 1 = drowsy, 0 = notdrowsy
    else:
        print("No training data found")
