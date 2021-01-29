
import cv2

image_path = '/Users/tieungao/Codes/python/ai-research/yolov5/test/01.png'

img = cv2.imread(image_path)

im0s = cv2.resize(img, (400, 400))

cv2.imwrite(image_path, im0s)