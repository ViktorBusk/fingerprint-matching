from PIL import Image, ImageOps

def load(image_path):
    I = Image.open(image_path)
    I = ImageOps.grayscale(I) 
    return I 
