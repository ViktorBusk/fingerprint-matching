from pathlib import Path
from image import Image
from feature_extraction import DirectionalField
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_path = Path(__file__).parents[1] / "database/UPEK/1_1.png"
    
    # Create image
    fingerprint_image = Image.open(image_path)
    directional_field = DirectionalField(fingerprint_image)
    