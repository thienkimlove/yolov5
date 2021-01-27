import argparse
import time

import pytesseract
from pathlib import Path
from datetime import datetime
import re
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import os
from reco.recognition import E2E
import time

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from csv import writer

def cleanup_image(img):
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

    string=(pytesseract.image_to_string(img))
    string=re.sub(r'\W+', '', string)
    return (string)


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


def match_license(lp_str):
    lp_str = re.sub(r'\W+', '', lp_str)
    regex = r"\d{2}[A-Z]{1}\d{4,5}"
    is_match = re.findall(regex, lp_str, re.MULTILINE)
    if not is_match:
        return None
    else:
        chars = " ".join(is_match)

        if len(chars) == 7:
            return "{} {}".format(chars[:3], chars[3:])
        else:
            return "{}-{}.{}".format(chars[:3], chars[3:6], chars[6:])


# this should be pixel instead of percent point 0-1
def get_crop_img_base_four(x_frame, x_min, y_min, x_max, y_max):
    return x_frame[int(y_min):int(y_max), int(x_min):int(x_max)]



# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def get_images_from_dir(dir_path, ext):
    lst_files = os.listdir(dir_path)
    lst_images = []

    for file in lst_files:
        if ext in file:
            lst_images.append("{}/{}".format(dir_path, file))
    return lst_images


def run_detect_on_img(
        img_path,
        vehicle_model_path,
        ano_file,
        current_count,
        reco_model,
        weights,
        gen_images_path,
        gen_crops_path
):
    # Load the model
    net = cv2.dnn.readNet(vehicle_model_path + '.xml', vehicle_model_path + '.bin')

    # Specify target device
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    iou_thres = 0.45
    imgsz = 640
    conf_thres = 0.6


    # filter by class: --class 0, or --class 0 2 3
    classes = 0
    agnostic_nms = True

    # Initialize
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    frame_origin = cv2.imread(img_path)
    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame_origin, size=(512, 512), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        label = float(detection[1])
        xmin = int(detection[3] * frame_origin.shape[1])
        ymin = int(detection[4] * frame_origin.shape[0])
        xmax = int(detection[5] * frame_origin.shape[1])
        ymax = int(detection[6] * frame_origin.shape[0])

        if confidence > conf_thres:
            im0s = get_crop_img_base_four(frame_origin, xmin, ymin, xmax, ymax)

            (h, w) = im0s.shape[:2]

            if h > 0 and w > 0:
                try:
                    im0s = cv2.resize(im0s, (400, 400))
                    img_detect = letterbox(im0s, new_shape=imgsz)[0]
                except Exception:
                    continue
            else:
                continue

            # Convert
            img_detect = img_detect[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img_detect = np.ascontiguousarray(img_detect)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            img = torch.from_numpy(img_detect).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=None)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred,
                conf_thres,
                iou_thres,
                classes=classes,
                agnostic=agnostic_nms
            )
            for i, det in enumerate(pred):
                # detections per image
                p, s, im0 = '', '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        lxmin = int(xyxy[0])
                        lymin = int(xyxy[1])
                        lxmax = int(xyxy[2])
                        lymax = int(xyxy[3])

                        im1s = get_crop_img_base_four(im0s, lxmin, lymin, lxmax, lymax)

                        try:
                            label1 = reco_model.predict(im0s)
                        except Exception:
                            label1 = None

                        label = cleanup_image(im1s)
                        reco_match = match_license(label1 or "")
                        tere_match = match_license(label or "")

                        if reco_match or tere_match:
                            display = reco_match if reco_match else tere_match
                            print("display={}".format(display))
                            current_count += 1


                            cv2.imwrite('{}/{}.png'.format(gen_images_path, current_count), im0s)
                            cv2.imwrite('{}/{}.png'.format(gen_crops_path, current_count), im1s)

                            # List
                            insert_str = [
                                current_count,
                                current_count,
                                '{}.png'.format(current_count),
                                display,
                                lxmin,
                                lymin,
                                lxmax,
                                lymax
                            ]

                            append_list_to_csv(ano_file, insert_str)
    return current_count


vehicle_model_path = '/Users/tieungao/Codes/python/ai-research/openvino-latest/models/downloaded/intel/vehicle-detection-0202/FP32/vehicle-detection-0202'
car_long_path = '/Users/tieungao/Codes/python/ai-research/openvino-latest/datasets/car_long'

car_gmt_path = '/Users/tieungao/Codes/python/ai-research/openvino-latest/datasets/CarTGMT'

ano_file = 'datav2/anov2_file.csv'

weights = 'weights/last_yolo5s_epoch500_time2_epoch1000_time3.pt'

gen_images_path = 'datav2/images'
gen_crops_path = 'datav2/crops'

reco_model = E2E()

images = get_images_from_dir(car_long_path, ".jpg")
current_count = 294
for image_path in images:
    current_count = run_detect_on_img(
        image_path,
        vehicle_model_path,
        ano_file,
        current_count,
        reco_model,
        weights,
        gen_images_path,
        gen_crops_path
    )

print(current_count)
