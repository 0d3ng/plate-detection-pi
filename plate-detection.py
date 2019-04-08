#  Created by od3ng on 08/04/2019 11:36:25 AM.
#  Project: plate-recognition-pi
#  File: plate-detection.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import cv2
import Utils
import sys
import os

# Folder untuk menyimpan dataset
path_slice = "dataset/sliced"
path_source = "dataset/source"

# Template untuk proyeksi vertikal
pv_template = Utils.proyeksi_vertical(cv2.imread("dataset/templates/plate/template.jpg", cv2.IMREAD_ANYCOLOR))

for file_name in sorted(os.listdir(path_source)):
    image = cv2.imread(os.path.join(path_source, file_name))
    src = image.copy()
    blurred = image.copy()
    print(image.shape)
    # Filtering gaussian blur
    for i in range(10):
        blurred = cv2.GaussianBlur(image, (5, 5), 0.5)

    # Conversi image BGR2GRAY
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image binerisasi menggunakan adaptive thresholding
    bw = cv2.adaptiveThreshold(rgb, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 10)

    # Operasi dilasi
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

    cv2.imwrite("bw.jpg", bw)
    cv2.namedWindow('bw', cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow('bw', bw)
    cv2.waitKey()

    # Ekstraksi kontur
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    slices = []
    img_slices = image.copy()
    idx = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ras = format(w / h, '.2f')
        # Pilih kontur dengan ukuran dan rasio tertentu
        if 30 <= h and (100 <= w <= 400) and (2.7 <= float(ras) <= 4):
            idx = idx + 1
            print("x={}, y={}, w={}, h={}, area={}, rasio={}".format(x, y, w, h, area, ras))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
            cv2.putText(image, "{}x{}".format(w, h), (x, int(y + (h / 2))), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            cv2.putText(image, "{}".format(ras), (x + int(w / 2), y + h + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255))
            crop = img_slices[y:y - 3 + h + 6, x:x - 3 + w + 6]
            slices.append(crop)

    cv2.imwrite("image.jpg", image)
    result = None
    max_value = sys.float_info.max
    for sl in slices:
        # cv2.imshow("slice ke {}".format(slices.index(sl) + 1), sl)
        # cv2.waitKey()
        pv_numpy = Utils.proyeksi_vertical(sl.copy())
        rs_sum = cv2.sumElems(cv2.absdiff(pv_template, pv_numpy))
        # print("sum: {} slice ke {}".format(rs_sum[0], slices.index(sl) + 1))
        if rs_sum[0] <= max_value:
            max_value = rs_sum[0]
            result = sl
    cv2.waitKey()
    cv2.imwrite(os.path.join(path_slice, file_name), result)
