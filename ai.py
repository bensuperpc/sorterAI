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

    def gpu(self):
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

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self._class_names

        return tf.argmax(one_hot)

    def decode_img(self, img):
        img = tf.io.decode_jpeg(img, dct_method="INTEGER_ACCURATE", channels=3)

        return tf.image.resize(img, [self._img_height, self._img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        return img, label

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self._batch_size)

        if self._data_augmentation:
            logger.debug("Using data augmentation")
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                # layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
                layers.RandomBrightness(0.1),
            ])
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=self._AUTOTUNE)

        ds = ds.prefetch(buffer_size=self._AUTOTUNE)
        return ds

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
            layers.Dense(len(self._class_names))
        ])

        #model.build((None, self._img_height, self._img_width, 3))
        # model.summary()

        return model

    def compile(self):
        self._model.compile(
            optimizer=self._optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    def load_model(self):
        self._model = keras.models.load_model(self._model_path)
        logger.debug(f"Model {self._model_path} loaded")

    def save_model(self):
        self._model.save(self._model_path, save_format='h5')
        logger.debug(f"Model saved to {self._model_path}")

    def prepare_train(self):
        logger.debug(f"Data loaded from {self._data_dir}")

        self._class_names = np.array(sorted(
            [item.name for item in self._data_dir.glob('*') if item.name != "LICENSE.txt"]))

        logger.debug(f"Class names: {self._class_names}")
        logger.debug(f"Number of classes: {len(self._class_names)}")

        logger.debug(
            f"Number of images: {len(list(self._data_dir.glob('*/*.jpg')))}")

        self._list_ds = tf.data.Dataset.list_files(str(self._data_dir/'*/*'))
        self._labeled_ds = self._list_ds.map(
            self.process_path, num_parallel_calls=self._AUTOTUNE)

        train_size = int(self._train_pourcent * len(self._labeled_ds))
        val_size = int(self._val_pourcent * len(self._labeled_ds))
        test_size = int(self._test_pourcent * len(self._labeled_ds))

        self._train_ds = self._labeled_ds.take(train_size)

        self._val_ds = self._labeled_ds.skip(train_size)
        self._val_ds = self._val_ds.take(val_size)

        self._test_ds = self._labeled_ds.skip(train_size + val_size)
        self._test_ds = self._test_ds.take(test_size)

        logger.debug(f"Train size: {len(self._train_ds)}")
        logger.debug(f"Val size: {len(self._val_ds)}")
        logger.debug(f"Test size: {len(self._test_ds)}")

        self._train_ds = self.configure_for_performance(self._train_ds)
        self._val_ds = self.configure_for_performance(self._val_ds)
        self._test_ds = self.configure_for_performance(self._test_ds)

        # Load model
        if self._model is None:
            logger.warning("Model is None, load default model")
            self._model = self.get_model()

            self._model.pop()
            self._model.add(layers.Dense(len(self._class_names)))
            #self._model.build((None, self._img_height, self._img_width, 3))
            # self._model.summary()

    def train(self):
        logger.debug("Start training")
        logger.debug(f"Epochs: {self._epochs}")
        self._history = self._model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=self._epochs
        )

    def evaluate(self):
        logger.debug("Start evaluation")
        loss, accuracy = self._model.evaluate(self._test_ds)

        logger.debug(f"Loss: {loss * 100} %")
        logger.debug(f"Accuracy: {accuracy * 100} %")

        return loss, accuracy

    # TODO : Need to be fixed
    def predict(self, img_path):
        logger.debug("Start prediction")
        # img = keras.preprocessing.image.load_img(
        #    img_path, target_size=(self._img_height, self._img_width)
        # )
        #img_array = keras.preprocessing.image.img_to_array(img)
        #img_array = tf.expand_dims(img_array, 0)

        image = cv.imread(img_path, 0)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.resize(image, (self._img_height, self._img_width))

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        predictions = self._model.predict(image_tensor)

        logger.debug(
            f"Predictions: {self._class_names[predictions]}")
        return predictions

    def display_history(self):
        acc = self._history.history['accuracy']
        val_acc = self._history.history['val_accuracy']

        loss = self._history.history['loss']
        val_loss = self._history.history['val_loss']

        epochs_range = range(self._epochs)

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
        plt.xlabel("{} {:2.0f}% ({})".format(self._class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             self._class_names[true_label]),
                   color=color)

    def plot_value_array(self, predictions_array, true_label, grid=False, pred_color='red', true_color='blue'):
        plt.grid(grid)
        plt.xticks(range(len(self._class_names)))
        plt.yticks([])
        thisplot = plt.bar(range(len(self._class_names)),
                           predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color(pred_color)
        thisplot[true_label].set_color(true_color)

    def display_predict(self):
        image_batch, label_batch = next(iter(self._test_ds))

        probability_model = tf.keras.Sequential([self._model,
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

    # Defining __init__ method
    def __init__(self):
        self.__name = "Not Set"
        logger.debug(f"TF version: {tf.__version__}")

        self._batch_size = 24
        self._img_height = 256
        self._img_width = 256
        self._epochs = 10

        self._AUTOTUNE = tf.data.AUTOTUNE

        self._class_names = []

        self._data_dir = None
        self._data_augmentation = False

        self._list_ds = None
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

        self._train_pourcent = 0.8
        self._val_pourcent = 0.1
        self._test_pourcent = 0.1

        self._history = None

        self._model = None
        # tf.keras.optimizers.Adam(learning_rate=1e-4)
        self._optimizer = "adam"
        self._model_path = "model.h5"

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def name(self, val):
        self._data_dir = val

    @name.deleter
    def data_dir(self):
        del self._data_dir

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def name(self, val):
        self._epochs = val

    @name.deleter
    def epochs(self):
        del self._epochs

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, val):
        self._batch_size = val

    @batch_size.deleter
    def batch_size(self):
        del self._batch_size


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-t', '--test', action='store_false')
    args = parser.parse_args()

    # Init AI
    ai = AI()

    # Enable GPU
    ai.gpu()

    data_dir = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    ai.data_dir = pathlib.Path(tf.keras.utils.get_file(
        'flower_photos', origin=data_dir, untar=True))

    logger.debug(f"Data dir: {ai.data_dir}")

    ai.prepare_train()
    ai.compile()

    ai.epochs = 9

    ai.train()

    ai.evaluate()

    if args.test:
        ai.display_predict()
        ai.display_history()
