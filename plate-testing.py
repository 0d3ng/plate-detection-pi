#  Created by od3ng on 10/04/2019 01:25:51 PM.
#  Project: plate-recognition-pi
#  File: plate-testing.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import numpy as np

data_dir_training = "dataset/training"
data_dir_testing = "dataset/testing"
dirs = []
width, height = 100, 100

model = tf.keras.models.load_model("anpr.model")

for char_name in sorted(os.listdir(data_dir_training)):
    dirs.append(char_name)

for car in sorted(os.listdir(data_dir_testing)):
    temp = ""
    for char_img in sorted(os.listdir(os.path.join(data_dir_testing, car))):
        img_array = cv2.imread(os.path.join(data_dir_testing, car, char_img), cv2.IMREAD_ANYCOLOR)
        new_array = cv2.resize(img_array, (width, height))
        new_array = np.array(new_array).reshape(-1, width, height, 1)

        new_array = new_array / 255.0

        prediction = model.predict(new_array)
        temp += dirs[np.argmax(prediction[0])]

    print("folder name: {} no: {}".format(car, temp))

    fig = plt.figure()
    fig.suptitle(temp, fontsize=16)
    columns = len(os.listdir(os.path.join(data_dir_testing, car)))
    rows = 1
    for i in range(1, columns * rows + 1):
        img = cv2.imread(os.path.join(data_dir_testing, car, str(i - 1) + ".jpg"))
        fig.add_subplot(rows, columns, i)
        plt.axis("off")
        plt.imshow(img, aspect='auto')
    plt.show()
