
#  Created by od3ng on 08/04/2019 11:36:14 AM.
#  Project: plate-recognition-pi
#  File: Utils.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def show_images(images):
    num = len(images)
    ax = np.ceil(np.sqrt(num))
    ay = np.rint(np.sqrt(num))
    fig = plt.figure()
    for i in range(1, num + 1):
        sub = fig.add_subplot(ax, ay, i)
        sub.axis('off')
        sub.imshow(images[i - 1])
    plt.show()


def proyeksi_vertical(img):
    blurred = cv2.GaussianBlur(img.copy(), (5, 5), 0)
    gray = cv2.cvtColor(blurred.copy(), cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey()
    resized = cv2.resize(gray.copy(), (450, 145))
    # cv2.imshow("resized", resized)
    # cv2.waitKey()
    ret, bw = cv2.threshold(resized.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("bw", bw)
    # cv2.waitKey()
    bw = bw / 255.
    bw_data = np.asarray(bw)
    pvertical = np.sum(bw_data, axis=1)
    return pvertical


def mse(image_a, image_b):
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err

def rotate_image(image, angle):
    '''Rotate image "angle" degrees.

    How it works:
      - Creates a blank image that fits any rotation of the image. To achieve
        this, set the height and width to be the image's diagonal.
      - Copy the original image to the center of this blank image
      - Rotate using warpAffine, using the newly created image's center
        (the enlarged blank image center)
      - Translate the four corners of the source image in the enlarged image
        using homogenous multiplication of the rotation matrix.
      - Crop the image according to these transformed corners
    '''

    diagonal = int(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))
    offset_x = int((diagonal - image.shape[0]) / 2)
    offset_y = int((diagonal - image.shape[1]) / 2)
    dst_image = np.zeros((diagonal, diagonal, 3), dtype='uint8')
    image_center = (diagonal / 2, diagonal / 2)

    R = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1]), :] = image
    dst_image = cv2.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv2.INTER_LINEAR)

    # Calculate the rotated bounding rect
    x0 = offset_x
    x1 = offset_x + image.shape[0]
    x2 = offset_x
    x3 = offset_x + image.shape[0]

    y0 = offset_y
    y1 = offset_y
    y2 = offset_y + image.shape[1]
    y3 = offset_y + image.shape[1]

    corners = np.zeros((3, 4))
    corners[0, 0] = x0
    corners[0, 1] = x1
    corners[0, 2] = x2
    corners[0, 3] = x3
    corners[1, 0] = y0
    corners[1, 1] = y1
    corners[1, 2] = y2
    corners[1, 3] = y3
    corners[2:] = 1

    c = np.dot(R, corners)

    x = int(c[0, 0])
    y = int(c[1, 0])
    left = x
    right = x
    up = y
    down = y

    for i in range(4):
        x = int(c[0, i])
        y = int(c[1, i])
        if x < left:
            left = x
        if x > right:
            right = x
        if y < up:
            up = y
        if y > down:
            down = y
    h = down - up
    w = right - left

    cropped = np.zeros((w, h, 3), dtype='uint8')
    cropped[:, :, :] = dst_image[left:(left + w), up:(up + h), :]
    return cropped


def noisy(noise_typ, image):
    '''

    :param noise_typ:str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    :param image:ndarray
    :return:
    '''
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


if __name__ == '__main__':
    src = cv2.imread("dataset/templates/plate/template.jpg", cv2.IMREAD_ANYCOLOR)
    print(src.shape)
    cv2.imshow("src", src)
    cv2.waitKey()
    cv2.imshow("gamma", adjust_gamma(src.copy(), 0.5))
    cv2.waitKey()
    for i in range(5, -5, -1):
        rotated = rotate_image(src.copy(), i)
        print(rotated.shape)
        cv2.imshow("rotate {} derajat".format(i), rotated)
        cv2.waitKey()
    # print(proyeksi_vertical(src))
