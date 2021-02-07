from pathlib import Path
from feature_extraction import OrientationField
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def RGB_to_gray_scale(I):
    #g = I.copy()
    I = cv2.cvtColor(np.array(I), cv2.COLOR_BGR2GRAY)
    return Image.fromarray(I)

if __name__ == '__main__':
    a = "database/UPEK/1_1.png"
    b = "database/FVC2004/DB1_B/101_1.tif"
    c = "database/AVA2017/AS.png"
    d = "database/AVA2017/city_test2.jpg"
    image_path = Path(__file__).parents[1] / a
    
    # Create image
    fingerprint_image = Image.open(image_path)
    #fingerprint_image = RGB_to_gray_scale(fingerprint_image)
    orientation_field = OrientationField(fingerprint_image, 10)
    orientation_field.show()