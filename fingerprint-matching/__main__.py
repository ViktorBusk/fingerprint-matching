from pathlib import Path
from image import load 
from feature_extraction import OrientationField

if __name__ == '__main__':
    a = "database/UPEK/1_2.png"
    b = "database/FVC2004/DB1_B/101_1.tif"
    c = "database/AVA2017/AS.png"
    d = "database/AVA2017/city_test2.jpg"
    image_path = Path(__file__).parents[1] / a
    
    # Create image
    fingerprint_image = load(image_path)
    orientation_field = OrientationField(fingerprint_image, 10)
    orientation_field.show()
