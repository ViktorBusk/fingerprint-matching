from pathlib import Path
from image import Image
from feature_extraction import DirectionalField

if __name__ == '__main__':
    image_path = Path(__file__).parents[1] / "database/UPEK/1_1.png"
    directional_field = DirectionalField(Image.open(image_path))
   
    # Testing
    # directional_field.plot_gradient()
    # sum_i_j = 0
    # for i in range(1, 4):
    #     for j in range(2, 5):
    #         sum_i_j += (i + j)

    # print(sum_i_j)