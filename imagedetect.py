import os
import numpy as np
from ultralytics import YOLO
import cv2


image = cv2.imread('test_images/construction-safety.jpg')


model = YOLO("best.pt")

results = model.predict(source=image, save=True, save_crop = True) 





