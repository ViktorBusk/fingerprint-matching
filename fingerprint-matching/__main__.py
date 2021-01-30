from pathlib import Path
from image import Image
from feature_extraction import DirectionalField

if __name__ == '__main__':
    image_path = Path(__file__).parents[1] / "database/UPEK/1_1.png" 
    directional_field = DirectionalField(Image.open(image_path))
    directional_field.plot_gradient()
   