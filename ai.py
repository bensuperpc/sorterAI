#!/usr/bin/env python

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from loguru import logger
import cv2 as cv


class AI:
    __author__ = "Bensuperpc"
    __copyright__ = None
    __credits__ = ["None", "None"]
    __license__ = "MIT"
    __version__ = "1.0.0"
    __maintainer__ = "Bensuperpc"
    __email__ = "bensuperpc@gmail.com"
    __status__ = "Development"
    __compatibility__ = ["Linux", "Windows", "Darwin"]
    __name__ = "AI"

    def __str__(self):
        return "AI"

    @staticmethod
    def gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                logger.debug(
                    f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                logger.error(e)

    @classmethod
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names

        return tf.argmax(one_hot)

    @classmethod
    def decode_img(self, img):
        img = tf.io.decode_jpeg(img, dct_method="INTEGER_ACCURATE", channels=3)

        return tf.image.resize(img, [self.img_height, self.img_width])

    @classmethod
    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        return img, label

    @classmethod
    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)

        if self.data_augmentation:
            logger.debug("Using data augmentation")
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                # layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
                layers.RandomBrightness(0.1),
            ])
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=self.AUTOTUNE)

        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    @classmethod
    def get_model(self):
        # data_augmentation = tf.keras.Sequential([
        #  layers.RandomFlip("horizontal_and_vertical"),
        #  layers.RandomRotation(0.2),
        # ])

        model = Sequential([
            # data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(len(self.class_names))
        ])

        #model.build((None, self.img_height, self.img_width, 3))
        # model.summary()

        return model

    @classmethod
    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    @classmethod
    def load_model(self):
        self.model = keras.models.load_model(self.model_path)
        logger.debug(f"Model {self.model_path} loaded")

    @classmethod
    def save_model(self):
        self.model.save(self.model_path, save_format='h5')
        logger.debug(f"Model saved to {self.model_path}")

    @classmethod
    def prepare_train(self):
        logger.debug(f"Loding data from {self.dataset_url}")

        self.data_dir = pathlib.Path(tf.keras.utils.get_file(
            'flower_photos', origin=self.dataset_url, untar=True))

        self.class_names = np.array(sorted(
            [item.name for item in self.data_dir.glob('*') if item.name != "LICENSE.txt"]))

        logger.debug(f"Class names: {self.class_names}")
        logger.debug(f"Number of classes: {len(self.class_names)}")

        logger.debug(
            f"Number of images: {len(list(self.data_dir.glob('*/*.jpg')))}")

        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
        self.labeled_ds = self.list_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)

        train_size = int(self.train_pourcent * len(self.labeled_ds))
        val_size = int(self.val_pourcent * len(self.labeled_ds))
        test_size = int(self.test_pourcent * len(self.labeled_ds))
        self.train_ds = self.labeled_ds.take(train_size)
        self.val_ds = self.labeled_ds.skip(train_size)
        self.val_ds = self.val_ds.take(val_size)
        self.test_ds = self.labeled_ds.skip(train_size + val_size)
        self.test_ds = self.test_ds.take(test_size)

        self.train_ds = self.configure_for_performance(self.train_ds)
        self.val_ds = self.configure_for_performance(self.val_ds)
        self.test_ds = self.configure_for_performance(self.test_ds)

        # Load model
        if self.model is None:
            logger.warning("Model is None, load default model")
            self.model = self.get_model()

            self.model.pop()
            self.model.add(layers.Dense(len(self.class_names)))
            #self.model.build((None, self.img_height, self.img_width, 3))
            # self.model.summary()

    @classmethod
    def train(self):
        logger.debug("Start training")
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

    @classmethod
    def evaluate(self):
        logger.debug("Start evaluation")
        loss, accuracy = self.model.evaluate(self.test_ds)

        logger.debug(f"Loss: {loss * 100} %")
        logger.debug(f"Accuracy: {accuracy * 100} %")

        return loss, accuracy

    # TODO : Need to be fixed
    @classmethod
    def predict(self, img_path):
        logger.debug("Start prediction")
        # img = keras.preprocessing.image.load_img(
        #    img_path, target_size=(self.img_height, self.img_width)
        # )
        #img_array = keras.preprocessing.image.img_to_array(img)
        #img_array = tf.expand_dims(img_array, 0)

        image = cv.imread(img_path, 0)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.resize(image, (self.img_height, self.img_width))

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        predictions = self.model.predict(image_tensor)

        logger.debug(
            f"Predictions: {self.class_names[predictions]}")
        return predictions

    @classmethod
    def display_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    @classmethod
    def plot_image(self, predictions_array, true_label, img, grid=False, pred_color='red', true_color='blue'):
        plt.grid(grid)
        plt.xticks([])
        plt.yticks([])
        img = np.array(img/np.amax(img)*255, np.int32)
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)

        if predicted_label == true_label:
            color = true_color
        else:
            color = pred_color
        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)

    @classmethod
    def plot_value_array(self, predictions_array, true_label, grid=False, pred_color='red', true_color='blue'):
        plt.grid(grid)
        plt.xticks(range(len(self.class_names)))
        plt.yticks([])
        thisplot = plt.bar(range(len(self.class_names)),
                           predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color(pred_color)
        thisplot[true_label].set_color(true_color)

    @classmethod
    def display_predict(self):
        image_batch, label_batch = next(iter(self.test_ds))

        probability_model = tf.keras.Sequential([self.model,
                                                tf.keras.layers.Softmax()])
        predictions = probability_model.predict(image_batch)

        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))

        for i in range(num_images):
            _label_batch = label_batch[i]
            _label_batch = _label_batch.numpy().tolist()

            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self.plot_image(predictions[i], _label_batch, image_batch[i])

            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self.plot_value_array(predictions[i], _label_batch)

        plt.tight_layout()
        plt.show()

    @classmethod
    def __init__(self, *args, **kwargs):
        logger.debug(f"TF version: {tf.__version__}")

        self.batch_size = 24
        self.img_height = 256
        self.img_width = 256
        self.epochs = 10

        self.AUTOTUNE = tf.data.AUTOTUNE

        self.class_names = []

        self.dataset_url = ""
        self.data_dir = None
        self.data_augmentation = False

        self.list_ds = None
        self.train_ds = None
        self.train_ds = None
        self.val_ds = None
        self.val_ds = None
        self.test_ds = None
        self.test_ds = None

        self.train_pourcent = 0.8
        self.val_pourcent = 0.1
        self.test_pourcent = 0.1

        self.history = None

        self.model = None
        # tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.optimizer = "adam"
        self.model_path = "model.h5"

        # Init GPU if available
        self.gpu()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-t', '--test', action='store_false')
    args = parser.parse_args()

    ai = AI()
    ai.dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    ai.prepare_train()
    ai.compile()
    ai.train()
    ai.evaluate()

    if args.test:
        ai.display_predict()
        ai.display_history()
