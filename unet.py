import tensorflow as tf
import keras
from keras.applications import MobileNetV3Small
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Dropout, UpSampling2D, Concatenate
from keras.models import Model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Функция для подготовки изображения
def prepare_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Нормализация
    image = np.expand_dims(image_array, axis=0)
    return image


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPool2D(2)(f)
    p = Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = Concatenate()([x, conv_features])
    # dropout
    x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def unet_model(input_shapes=(128, 128, 3)):
    # inputs
    inputs = Input(shape=input_shapes)
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = Conv2D(3, 1, padding="same", activation="softmax")(u9)
    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")
    return unet_model


# def unet_model(input_shape):
#     inputs = Input(shape=input_shape)
#
#     # Энкодер (используем предобученный MobileNetV2)
#     base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=inputs)
#
#     # Извлекаем выходы из блоков MobileNetV3
#     conv1 = base_model.get_layer('expanded_conv/project/BatchNorm').output
#     conv2 = base_model.get_layer('expanded_conv_2/project/BatchNorm').output
#     conv3 = base_model.get_layer('expanded_conv_5/project/BatchNorm').output
#     conv4 = base_model.get_layer('expanded_conv_11/project/BatchNorm').output
#     conv5 = base_model.get_layer('expanded_conv_17/project/BatchNorm').output
#
#     # Создаем блоки деконволюции
#     up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
#     merge6 = Concatenate([conv4, up6])
#     conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
#     conv6 = Dropout(0.2)(conv6)
#
#     up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
#     merge7 = Concatenate([conv3, up7])
#     conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
#     conv7 = Dropout(0.2)(conv7)
#
#     up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
#     merge8 = Concatenate([conv2, up8])
#     conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
#     conv8 = Dropout(0.2)(conv8)
#
#     up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
#     merge9 = Concatenate([conv1, up9])
#     conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
#     conv9 = Dropout(0.2)(conv9)
#
#     # Выходной слой
#     output = Conv2D(3, (1, 1), padding="same", activation="softmax")(conv9)
#
#     model = Model(inputs=base_model.input, outputs=output)
#     return model


input_shape = (128, 128, 3)
model = unet_model(input_shape)
print(model.summary())
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# image_path = './111.png'
#
# # Подготовка изображения
# image = prepare_image(image_path)
#
# # Получение предсказания
# prediction = model.predict(image)
# print(prediction)
# # Преобразование предсказания в маску
# predicted_mask = (prediction > 0.5).astype(np.float32)
# print(predicted_mask)
# # Визуализация результатов
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(image[0])
