from pathlib import Path
from feature_extraction import OrientationField
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = "database/UPEK/1_1.png"
    b = "database/FVC2004/DB1_B/101_1.tif"
    image_path = Path(__file__).parents[1] / a
    
    # Create image
    fingerprint_image = Image.open(image_path)
    orientation_field = OrientationField(fingerprint_image, 12)
    orientation_field.show()