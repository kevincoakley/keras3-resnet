import os
import argparse
import math
import sys
import csv
from datetime import datetime


def get_dataset_details(dataset_name):
    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cifar10": {
            "train_path": "./cifar10/train/",
            "val_path": "./cifar10/val/",
            "test_path": "./cifar10/test/",
            "num_classes": 10,
            "dataset_shape": (32, 32, 3),
            "normalization": {
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.247, 0.2435, 0.2616),
            },
        },
        "cifar100": {
            "train_path": "./cifar100/train/",
            "val_path": "./cifar100/val/",
            "test_path": "./cifar100/test/",
            "num_classes": 100,
            "dataset_shape": (32, 32, 3),
            "normalization": {
                "mean": (0.5071, 0.4865, 0.4409),
                "std": (0.2673, 0.2564, 0.2762),
            },
        },
    }

    return datasets[dataset_name]


def train(
    backend,
    data_loader,
    op_determinism,
    seed_val,
    model_name,
    dataset_name,
    epochs,
    lr_warmup,
):

    # Set the backend for the keras
    os.environ["KERAS_BACKEND"] = backend

    #
    # Set the XLA flags for JAX backend
    #
    XLA_FLAGS = ""

    # Set the XLA flags for JAX backend for deterministic operations
    if op_determinism and backend == "jax":
        XLA_FLAGS = XLA_FLAGS + "--xla_gpu_deterministic_ops=true"

    if data_loader == "torch" and backend == "jax":
        XLA_FLAGS = XLA_FLAGS + " --xla_gpu_enable_command_buffer="

    if XLA_FLAGS != "":
        os.environ["XLA_FLAGS"] = XLA_FLAGS

    import keras
    import tensorflow as tf
    import resnet_conv4

    # Set the random seed for reproducibility
    if seed_val != 0:
        keras.utils.set_random_seed(seed_val)

    # Set the deterministic operations for the tensorflow and torch backends
    if op_determinism:
        if backend == "tensorflow":
            import tensorflow as tf

            tf.config.experimental.enable_op_determinism()
        elif backend == "torch":
            import torch

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

    # Set the batch size
    batch_size = 128

    # Get the dataset details
    dataset_details = get_dataset_details(dataset_name)
    dataset_shape = dataset_details["dataset_shape"]
    num_classes = dataset_details["num_classes"]

    #
    # Load the dataset using either the tensorflow or torch data loader
    #
    if data_loader == "tensorflow":
        import data_loader_tensorflow

        train_dataset, val_dataset, test_dataset = data_loader_tensorflow.load_dataset(
            dataset_details, batch_size, seed_val
        )
    elif data_loader == "torch":
        import data_loader_pytorch

        train_dataset, val_dataset, test_dataset = data_loader_pytorch.load_dataset(
            dataset_details, batch_size, seed_val
        )

    # Define the models
    model_functions = {
        "ResNet20": resnet_conv4.resnet20,
        "ResNet32": resnet_conv4.resnet32,
        "ResNet44": resnet_conv4.resnet44,
        "ResNet56": resnet_conv4.resnet56,
        "ResNet110": resnet_conv4.resnet110,
        "ResNet1202": resnet_conv4.resnet1202,
    }

    # Create the model
    model = model_functions[model_name](
        input_shape=dataset_shape, num_classes=num_classes
    )

    # Print the model summary
    model.summary()

    # Print the current backend
    print("Keras backend:", keras.backend.backend())

    # Set the learning rate
    learning_rate = 0.1

    # Define the learning rate schedule
    def lr_schedule(epoch):
        if lr_warmup and epoch < 5:
            return learning_rate * 0.1
        elif epoch < math.ceil(epochs * 0.5):
            return learning_rate
        elif epoch < math.ceil(epochs * 0.75):
            return learning_rate * 0.1
        else:
            return learning_rate * 0.01

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.SGD(
            weight_decay=0.0001,
            momentum=0.9,
            learning_rate=lr_schedule(0),
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Define the learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # Define the callbacks
    callbacks = [lr_scheduler]

    # Time the training
    start_time = datetime.now()

    # Train the model
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    # Calculate the training time
    end_time = datetime.now()
    training_time = end_time - start_time

    # Evaluate the model
    score = model.evaluate(test_dataset, verbose=0)
    
    # Print the test loss, test accuracy, and training time
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Training time:", int(training_time.total_seconds()))

    return score[0], score[1], training_time


def save_score(
    test_loss,
    test_accuracy,
    training_time,
    backend,
    data_loader,
    model_name,
    dataset_name,
    op_determinism,
    seed_val,
    epochs,
    lr_warmup,
    filename,
):
    csv_file = filename + ".csv"
    write_header = False

    # If determistic is false and the seed value is 1 then the
    # seed value is totally random and we don't know what it is.
    if seed_val == 0:
        seed_val = "random"

    if not os.path.isfile(csv_file):
        write_header = True

    with open(csv_file, "a") as csvfile:
        fieldnames = [
            "date_time",
            "backend",
            "data_loader",
            "model_name",
            "dataset_name",
            "fit_time",
            "epochs",
            "lr_warmup",
            "op_determinism",
            "random_seed",
            "test_loss",
            "test_accuracy",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "date_time": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                "backend": backend,
                "data_loader": data_loader,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "fit_time": int(training_time.total_seconds()),
                "epochs": epochs,
                "lr_warmup": lr_warmup,
                "op_determinism": op_determinism,
                "random_seed": seed_val,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--op-determinism",
        dest="op_determinism",
        help="Run with deterministic operations",
        action="store_true",
    )

    parser.add_argument(
        "--seed-val", dest="seed_val", help="Set the seed value", type=int, default=0
    )

    parser.add_argument(
        "--data-loader",
        dest="data_loader",
        help="Data loader to use for training",
        default="tensorflow",
        choices=[
            "torch",
            "tensorflow",
        ],
        required=True,
    )

    parser.add_argument(
        "--backend",
        dest="backend",
        help="Backend to use for training",
        default="tensorflow",
        choices=[
            "jax",
            "torch",
            "tensorflow",
        ],
        required=True,
    )

    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Use the learning rate warmup of 5 epochs",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        help="Number of epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="The dataset to train the model on",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
        ],
        required=True,
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="Name of model to train",
        default="ResNet20",
        choices=[
            "ResNet20",
            "ResNet32",
            "ResNet44",
            "ResNet56",
            "ResNet110",
            "ResNet1202",
        ],
        required=True,
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    test_loss, test_accuracy, training_time = train(
        backend=args.backend,
        data_loader=args.data_loader,
        op_determinism=args.op_determinism,
        seed_val=args.seed_val,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        lr_warmup=args.lr_warmup,
    )

    save_score(
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        training_time=training_time,
        backend=args.backend,
        data_loader=args.data_loader,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        op_determinism=args.op_determinism,
        seed_val=args.seed_val,
        epochs=args.epochs,
        lr_warmup=args.lr_warmup,
        filename="train_results",
    )
