import cv2
import numpy as np
import ivate.pyplot as plt
import math
img = cv2.imread('./datasets/test_img/DLMP005.jpg')
edges = cv2.Canny(img, 100, 200)
