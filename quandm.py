import PIL
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
import tensorflow as tf
from tensorflow_core.python.training import monitored_session

import tensor
from ocr_models import common_flags, datasets, data_provider
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


vehicle_model_path = '/Users/tieungao/Codes/python/ai-research/openvino-latest/models/downloaded/intel/vehicle-detection-0202/FP32/vehicle-detection-0202'

# Load the model
net = cv2.dnn.readNet(vehicle_model_path + '.xml', vehicle_model_path + '.bin')

video_path = '/Users/tieungao/Codes/python/ai-research/openvino-latest/samples/sample-videos/pixabay/sg_street.mp4'

# tensorflow ano

ano_file = 'gen_datav2/annot_file.csv'

reco_model = E2E()

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out_video = None

countStep = 0

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print("Number of frames: ", video_length)
count = 0
print("Converting video..\n")

iou_thres = 0.45
view_img = False
imgsz = 640
conf_thres = 0.8
weights = 'weights/last_yolo5s_epoch500_time2_epoch1000_time3.pt'

# filter by class: --class 0, or --class 0 2 3
classes = 0
agnostic_nms = True

# using default img_size
webcam = False
# Initialize
device = select_device()
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16
number_unique = 0


batch_size = 1
dataset_name = 'quandm'
checkpoint = '/Users/tieungao/Codes/python/ai-research/yolov5/ocr52/model.ckpt-60404'
tf.compat.v1.reset_default_graph()
images_placeholder, endpoints = tensor.create_model(batch_size, dataset_name)

session_creator = monitored_session.ChiefSessionCreator(
  checkpoint_filename_with_path=checkpoint)
sess = monitored_session.MonitoredSession(
      session_creator=session_creator)


while cap.isOpened():
    ret, frame_origin = cap.read()
    if not ret:
        break  # abandons the last frame in case of async_mode

    if not out_video:
        out_video = cv2.VideoWriter('result/output2.avi', fourcc, 20.0, (frame_origin.shape[1], frame_origin.shape[0]))

    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame_origin, size=(512, 512), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    # Draw detected faces on the frame
    # The net outputs blob with shape: [1, 1, N, 7],
    # where N is the number of detected bounding boxes.
    # Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max],

    for detection in out.reshape(-1, 7):
        # print(detection)
        confidence = float(detection[2])
        label = float(detection[1])
        xmin = int(detection[3] * frame_origin.shape[1])
        ymin = int(detection[4] * frame_origin.shape[0])
        xmax = int(detection[5] * frame_origin.shape[1])
        ymax = int(detection[6] * frame_origin.shape[0])

        if confidence > 0.8:
            # print(detection)

            im0s = get_crop_img_base_four(frame_origin, xmin, ymin, xmax, ymax)

            # cv2.rectangle(frame_origin, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

            (h, w) = im0s.shape[:2]

            if h > 0 and w > 0:
                try:
                    imgDetect = letterbox(im0s, new_shape=imgsz)[0]
                except Exception:
                    continue
            else:
                continue

            # Convert
            imgDetect = imgDetect[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imgDetect = np.ascontiguousarray(imgDetect)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            t0 = time.time()

            img = torch.from_numpy(imgDetect).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=None)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes,
                                       agnostic=agnostic_nms)
            t2 = time_synchronized()

            have_license = False

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = '', '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):

                    have_license = True
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f'{n} {names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # label = f'{names[int(cls)]} {conf:.2f}'

                        lxmin = int(xyxy[0])
                        lymin = int(xyxy[1])
                        lxmax = int(xyxy[2])
                        lymax = int(xyxy[3])

                        # create image from xmin, ymin and xmax, ymax

                        fxmin = lxmin + int(xmin)
                        fxmax = lxmax + int(xmin)

                        fymin = lymin + int(ymin)
                        fymax = lymax + int(ymin)

                        # correct
                        # im1s = frame_origin[int(fymin):int(fymax), int(fxmin):int(fxmax)]
                        # im1s = im0s[lymin:lymax, lxmin:lxmax]

                        im1s = get_crop_img_base_four(im0s, lxmin, lymin, lxmax, lymax)

                        # cv2.rectangle(frame_origin, (fxmin, fymin), (fxmax, fymax), color=(0, 255, 0))
                        # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        # im0s = frame_origin[int(ymin):int(ymax), int(xmin):int(xmax)]

                        detect_box = [
                            fxmin,  # xmin
                            fymin,  # ymin
                            fxmax,
                            fymax
                        ]

                        try:
                            label1 = reco_model.predict(im0s)
                        except Exception:
                            label1 = None

                        im0s2 = cv2.resize(im0s, (400, 400))

                        # label2 = tensorflow_run(im0s2)

                        images_data = tensor.load_images(im0s2, batch_size, dataset_name)

                        predictions = sess.run(endpoints.predicted_text,
                                               feed_dict={images_placeholder: images_data})

                        # print(images_placeholder)
                        # print(images_data)

                        # op = sess.graph.get_operations()
                        # for o in op:
                        #     print(o.name)
                        is_list = [pr_bytes.decode('utf-8') for pr_bytes in predictions.tolist()]

                        label2 = " ".join(is_list)

                        # gray = get_grayscale(im1s)
                        # thresh_img = thresholding(gray)
                        # opening_img = opening(gray)
                        # canny_img = canny(gray)
                        # custom_config = r'--oem 3 --psm 6'
                        # label_origin = pytesseract.image_to_string(im1s, config=custom_config)
                        # label_thresh = pytesseract.image_to_string(thresh_img, config=custom_config)
                        #
                        # label_opening = pytesseract.image_to_string(opening_img, config=custom_config)
                        # label_canny = pytesseract.image_to_string(canny_img, config=custom_config)
                        #
                        # label = '{}|{}|{}|{}'.format(label_origin, label_thresh, label_opening, label_canny)
                        reco_detect = "reco detect = {}".format(label1)
                        tensor_detect = "tensor detect = {}".format(label2)
                        label = cleanup_image(im1s)
                        reco_match = match_license(reco_detect)
                        tere_match = match_license(label or "")
                        tensor_match = match_license(label2 or "")

                        if reco_match or tere_match or tensor_match:
                            display = reco_match if reco_match else tensor_match if tensor_match else tere_match
                            print("display={}".format(display))

                            if tensor_match:
                                print("From tensor_match")

                            # to 200x200

                            # resized = cv2.resize(im1s, (200, 200), interpolation=cv2.INTER_AREA)
                            # cv2.imshow("resized", resized)

                            count += 1
                            cv2.imwrite('gen_datav2/images/{}.png'.format(count), im0s)
                            cv2.imwrite('gen_datav2/crops/{}.png'.format(count), im1s)

                            # List
                            insert_str = [count, count, '{}.png'.format(count), display, lxmin, lymin, lxmax, lymax]

                            append_list_to_csv(ano_file, insert_str)

                            # tensorflow format
                            # , Unnamed: 0, files, text, xmin, ymin, xmax, ymax
                            # 0, 0, 0.png, MH15 - TC - 554, 589, 280, 694, 311

                            plot_one_box(detect_box, frame_origin, label=display, color=colors[int(cls)], line_thickness=3)

            # if have_license:
            #     number_unique += 1
            #     cv2.imwrite('temp/{}.jpg'.format(number_unique), im0s)

    # cv2.imshow('Detection Results', frame_origin)
    # key = cv2.waitKey(1)
    #
    # ESC_KEY = 27
    # # Quit.
    # if key in {ord('q'), ord('Q'), ESC_KEY}:
    #     break

    # Write the results back to output location.
    # cv2.imwrite("result/%#05d.jpg" % (count + 1), frame_origin)

    # write the flipped frame
    out_video.write(frame_origin)

# Release everything if job is finished
cap.release()
out_video.release()
cv2.destroyAllWindows()
