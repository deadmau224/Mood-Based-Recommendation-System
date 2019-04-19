import cv2
import numpy as np

image = cv2.imread('/Users/digvijayghotane/Desktop/Projects/mood_based_recommendation_system/Beta/image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("image.png",gray)