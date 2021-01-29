import tensorflow as tf

import cv2
from csv import writer
import os, sys, re
from pytesseract import pytesseract
import numpy as np
import csv

def cleanup_image_and_get_string(img):
    #Resize the image with interpolation
    # cv2.imwrite("result/test/number_plate_original_{}.png".format(count), img)
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite("result/test/number_plate_resized_{}.png".format(count), img)
    #Convert to Grey scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("result/test/number_plate_greyed_{}.png".format(count), img)

    #Define the kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    # Dilation expands whitespace breaking up characters and removing noise
    img = cv2.dilate(img, kernel, iterations=1)
    #cv2.imwrite("result/test/number_plate_dilated_{}.png".format(count), img)

    # Erosion makes the dark characters bigger

    img = cv2.erode(img, kernel, iterations=1)
    #cv2.imwrite("result/test/number_plate_eroded_{}.png".format(count), img)

    string = (pytesseract.image_to_string(img))
    string = re.sub(r'\W+', '', string)
    return (string)


def get_images_from_dir(dir_path, ext):
    lst_files = os.listdir(dir_path)
    lst_images = []

    for file in lst_files:
        if ext in file:
            lst_images.append("{}/{}".format(dir_path, file))
    return lst_images


def append_list_to_csv(path_csv, insert_list):
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(path_csv, 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)

        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(insert_list)

        # Close the file object
        f_object.close()


img_dir = 'gen_data/crops'

ano_file = 'gen_data/ano.csv'

data_dir = 'datav3'

new_train_txt = 'datav3/annotations-training.txt'
new_test_txt = 'datav3/annotations-testing.txt'


with open(ano_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if i > 0:
            lst = line[0].split(',')
            number_order = lst[1]
            label = lst[3]
            label = re.sub(r'\W+', '', label)
            if int(number_order) < 700:
                append_list_to_csv(new_train_txt, ['gen_data/crops/{}.png {}'.format(number_order, label)])

            else:
                append_list_to_csv(new_test_txt, ['gen_data/crops/{}.png {}'.format(number_order, label)])

# list_images = get_images_from_dir(img_dir, '.png')
#
# for list_image in list_images:
#     number_order = list_image.replace('.png', '')
#     number_order = number_order.replace('gen_data/crops/', '')
#
#     if int(number_order) < 700:
#         print("those for train...")
#         insert_str = [
#             list_image,
#             current_count,
#             '{}.png'.format(current_count),
#             display,
#             lxmin,
#             lymin,
#             lxmax,
#             lymax
#         ]
#
#         append_list_to_csv(ano_file, insert_str)

