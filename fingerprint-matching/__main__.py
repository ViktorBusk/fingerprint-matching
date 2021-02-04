from pathlib import Path
from feature_extraction import OrientationField
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_path = Path(__file__).parents[1] / "database/UPEK/1_1.png"
    
    # Create image
    fingerprint_image = Image.open(image_path)
    orientation_field = OrientationField(fingerprint_image)
    orientation_field.show()