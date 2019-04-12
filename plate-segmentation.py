#  Created by od3ng on 09/04/2019 09:55:12 AM.
#  Project: plate-recognition-pi
#  File: plate-segmentation.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import cv2
import os
import Utils

# Lokasi hasil pelat
path_plate = "dataset/sliced"
data_dir_testing = "dataset/testing"

# Looping file di direktori
for name_file in sorted(os.listdir(path_plate)):
    src = cv2.imread(os.path.join(path_plate, name_file))
    blurred = src.copy()
    gray = blurred.copy()

    # Filtering
    for i in range(10):
        blurred = cv2.GaussianBlur(src, (5, 5), 0.5)

    # Ubah ke grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Image binary
    ret, bw = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(ret, bw.shape)
    cv2.imwrite("segmentasi-bw.jpg", bw)
    # cv2.imshow("bw", bw)
    # cv2.waitKey()

    # Image morfologi, opening
    erode = cv2.erode(bw.copy(), cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3)))
    sliced = erode.copy()
    cv2.imwrite("segmentasi-erode.jpg", erode)
    # cv2.imshow("erode", erode)
    # cv2.waitKey()

    # Ekstraksi kontur
    contours, hierarchy = cv2.findContours(erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours_temps = []

    # Looping contours untuk mendapatkan kontur yang sesuai
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ras = format(w / h, '.2f')
        # print("x={}, y={}, w={}, h={}, rasio={}".format(x, y, w, h, ras))
        if h >= 40 and w >= 10 and float(ras) <= 1:
            # Gambar segiempat hasil segmentasi warna merah
            h += 1
            w += 1
            cv2.rectangle(src, (x - 1, y - 1), (x + w, y + h), (0, 0, 255), thickness=1)
            print("+ x={}, y={}, w={}, h={}, rasio={}".format(x, y, w, h, ras))
            contours_temps.append(cnt)
    cv2.imwrite("segmentasi-result.jpg", src)

    # Sorting contour terlebih dahulu agar karakter pembacaan dari kiri ke kanan
    contours, bounding_boxes = Utils.sort_contours(contours_temps)

    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        h += 1
        w += 1
        # Buat direktori berdasarkan nama file
        if not os.path.exists(os.path.join(data_dir_testing, os.path.splitext(name_file)[0])):
            os.mkdir(os.path.join(data_dir_testing, os.path.splitext(name_file)[0]))
        crop = sliced[y - 1:y + h, x - 1:x + w]
        cv2.imwrite(os.path.join(data_dir_testing, os.path.splitext(name_file)[0], str(idx) + ".jpg"), crop)
        idx += 1

    # cv2.imshow("result", src)
    # cv2.waitKey()
