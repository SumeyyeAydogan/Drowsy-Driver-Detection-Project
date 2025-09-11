import matplotlib.pyplot as plt
import tensorflow as tf

# örnek augment bloğu (senin data_augmentation değişkenin)
rotation_factor = 0.30
zoom_factor      = 0.30
brightness_delta = 0.1
contrast_factor  = 0.30

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(rotation_factor),
    tf.keras.layers.RandomZoom(zoom_factor),
    tf.keras.layers.RandomBrightness(brightness_delta,
                                     value_range=(0.0, 1.0)),
    tf.keras.layers.RandomContrast(contrast_factor)
])

# Görsel örnekleme (tek batch)
sample_ds = tf.keras.utils.image_dataset_from_directory(
    "data",       # senin klasörün
    labels="inferred",
    label_mode="int",
    image_size=(224, 224),
    batch_size=8,       # küçük olsun ki görmesi kolay olsun
    shuffle=True
)

# bir batch al
for images, labels in sample_ds.take(1):
    augmented_images = data_augmentation(images)
    plt.figure(figsize=(12, 6))
    for i in range(8):
        ax = plt.subplot(2, 8, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
        ax = plt.subplot(2, 8, i+9)
        plt.imshow(augmented_images[i].numpy())
        plt.axis("off")
    plt.show()
