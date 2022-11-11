from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def download_files(paths):
    data = []
    labels = []
    for path in paths:
        label = path.split(os.path.sep)[-2]

        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)
    return data, labels



# консты скорость, кол-во картинок. кол-во данных за ход
SPEED = 0.004
QUANTITY = 101
BS = 32


# че куда читать считывать
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to output face mask detector model")
args = vars(ap.parse_args())

paths = list(paths.list_images(args["dataset"]))
# загрузим data
data, labels = download_files(paths)
# переведем data в нумпай таблицу
data = np.array(data, dtype="float32")
labels = np.array(labels)


# бинарка 0-без 1-с маской
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# тут при помощи либы Scikit-learn рандомируем в адекватном кол-ве нулей и единиц в train и test
# на обучение оставим 80% данных
trainX, testX, trainY, testY = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# добавляем перевернутые фотографии
newData = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# создадим базовую модель из уже обученной готовой архитектуры с весами
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# магия..
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# базовая модель принимает внешний слой
model = Model(inputs=baseModel.input, outputs=headModel)

# Заморозить все слои предварительно обученной модели
for layer in baseModel.layers:
    layer.trainable = False

# скомилируем модель
print("[INFO] compiling model...")
opt = Adam(lr=SPEED, decay=SPEED / QUANTITY)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# тренировка
print("[INFO] training head...")
H = model.fit(
    newData.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=QUANTITY)

# BS(кол-во) результатов
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# берем с наиб показателем
predIdxs = np.argmax(predIdxs, axis=1)

# вывести отчет обучения
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

#
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")
