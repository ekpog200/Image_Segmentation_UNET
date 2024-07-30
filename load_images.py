import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd


base_dir = './datasets/oxford-iiit-pet/'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'annotations', 'trimaps')


def load_image(image_path, mask_path, img_size=(128, 128)):
    """
    :param image_path: путь к изображению
    :param mask_path: путь к маске
    :param img_size: размер, к которому преобразуют изображение/маску
    :return:
    """
    # Загрузка/преобразование в тензор 3 канала, resize, преобразование пикселей к [0,1]
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    # Загрузка маски с 1 каналом
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask


@tf.function
def random_flip(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask


@tf.function
def random_brightness(image, mask):
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, mask


@tf.function
def random_contrast(image, mask):
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, mask


@tf.function
def augment(image, mask):
    image, mask = random_flip(image, mask)
    image, mask = random_brightness(image, mask)
    image, mask = random_contrast(image, mask)
    return image, mask


def create_dataset(image_dir, mask_dir, img_size=(128, 128)):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    df = pd.DataFrame({
        'image_path': [os.path.join(image_dir, f) for f in image_files],
        'mask_path': [os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in image_files]
    })
    # преобразование df в tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, df['mask_path'].values))
    # изменение параметров изображения и маски
    return dataset.map(lambda x, y: load_image(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def pipeline_load_images():

    # Создание полного датасета
    full_dataset = create_dataset(image_dir, mask_dir)

    # Разделение на train и test
    train_size = int(0.8 * len(os.listdir(image_dir)))

    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)

    batch_size = 16  # кол-во батчей
    buffer_size = 1000  # кол-во элементов в памяти для перемешивания

    # пайплайн для train данных
    train_batches = (train_dataset
                     .map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # аугментация
                     .cache()  # кеширование для ускорения
                     .shuffle(buffer_size)  # перемешивания
                     .batch(batch_size)  # деление на батчи
                     .repeat()  # повторение данных
                     .prefetch(buffer_size=tf.data.AUTOTUNE))  # предзагрузка в фоновом решиме
    # Разделение test на validation и test
    validation_size = 3000
    test_size = 669

    validation_batches = test_dataset.take(validation_size).batch(batch_size)
    test_batches = test_dataset.skip(validation_size).take(test_size).batch(batch_size)

    sample_batch = next(iter(train_batches))
    random_index = np.random.choice(sample_batch[0].shape[0])
    sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
    display([sample_image, sample_mask])
    return train_batches, validation_batches, test_batches
