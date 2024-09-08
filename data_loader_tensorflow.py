import tensorflow as tf


def load_dataset(dataset_details, batch_size, seed_val=0):

    # Set seed for reproducibility
    if seed_val != 0:
        tf.keras.utils.set_random_seed(seed_val)

    # Get the dataset details
    train_path = dataset_details["train_path"]
    val_path = dataset_details["val_path"]
    test_path = dataset_details["test_path"]
    dataset_shape = dataset_details["dataset_shape"]
    normalization_mean = dataset_details["normalization"]["mean"]
    normalization_std = dataset_details["normalization"]["std"]

    #
    # Define the preprocessing and augmentation transformations
    #
    def preprocessing(image, label):
        image = tf.cast(image, tf.float32)
        # Normalize the pixel values ((input[channel] - mean[channel]) / std[channel])
        image = tf.divide(
            image, (255.0, 255.0, 255.0)
        )  # divide by 255 to match pytorch
        image = tf.subtract(image, normalization_mean)
        image = tf.divide(image, normalization_std)
        return image, label

    def augmentation(image, label):
        # Pad the image by 25% of the dataset shape
        image = tf.image.resize_with_crop_or_pad(
            image,
            round(dataset_shape[0] + (dataset_shape[0] * 0.25)),
            round(dataset_shape[1] + (dataset_shape[1] * 0.25)),
        )
        # Randomly crop the image to the dataset shape
        image = tf.image.random_crop(image, dataset_shape)
        # Randomly flip the image horizontally
        image = tf.image.random_flip_left_right(image)
        return image, label

    #
    # Get the training, validation, and test datasets from the directory
    #
    train = tf.keras.utils.image_dataset_from_directory(
        train_path,
        shuffle=True,
        image_size=dataset_shape[:2],
        interpolation="nearest",
        batch_size=None,
    )
    val = tf.keras.utils.image_dataset_from_directory(
        val_path,
        shuffle=False,
        image_size=dataset_shape[:2],
        interpolation="nearest",
        batch_size=None,
    )
    test = tf.keras.utils.image_dataset_from_directory(
        test_path,
        shuffle=False,
        image_size=dataset_shape[:2],
        interpolation="nearest",
        batch_size=None,
    )

    print(f"Number of training samples: {train.cardinality()}")
    print(f"Number of validation samples: {val.cardinality()}")
    print(f"Number of test samples: {test.cardinality()}")

    #
    # Batch and prefetch the dataset
    #
    train_dataset = (
        train.map(preprocessing)
        .map(augmentation)
        .shuffle(1000)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
