import os
import cv2 as cv
import asyncio

from dotenv import load_dotenv
from core.yolo_detect import YoloDetect


load_dotenv()
camera_index = os.environ.get('CAMERA_INDEX')

capture = cv.VideoCapture(camera_index)

labels = []
model_config = './data/yolov3.cfg'
model_weights = './data/yolov3.weights'
net = cv.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

with open('./data/coco.names', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

asyncio.run(YoloDetect(capture, labels, net))
cv.destroyAllWindows()
