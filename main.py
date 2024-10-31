import argparse
import sys

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


model = 'efficientdet_lite0.tflite'
camera_id = 0
width = 640
height = 480
num_threads = 4

cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

base_options = core.BaseOptions(file_name=model, num_threads=num_threads)

detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)

options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

detector = vision.ObjectDetector.create_from_options(options)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    detection_result = detector.detect(input_tensor)
    print(detection_result)
    
    if cv2.waitKey(1) == 27:
        break
