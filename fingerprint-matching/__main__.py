from pathlib import Path
from image import Image
from feature_extraction import DirectionalField
import numpy as np
import matplotlib.pyplot as plt

def M(I):
    return 1 / (I.shape[0]*I.shape[1]) * np.sum(I)

def VAR(I):
    sum = 0
    MI = M(I)
    for i in range(I.shape[1]):
        for j in range(I.shape[0]):
            sum += ((I[j, i] - MI)**2)
    return 1 / (I.shape[0]*I.shape[1]) * sum

if __name__ == '__main__':
    image_path = Path(__file__).parents[1] / "database/UPEK/1_1.png"
    
    # Create image
    image = Image.open(image_path)
    I = np.array(image)
    g = np.empty(I.shape)
    
    # Normalize image
    M0, VAR0 = 100, 100
    M, VAR = M(I), VAR(I)

    for i in range(I.shape[1]):
        for j in range(I.shape[0]):
            if  I[j, i] > M:
                g[j, i] = (M0 + np.sqrt(VAR0 * (I[j, i] - M)**2 / VAR))
            else:
                g[j, i] = (M0 - np.sqrt(VAR0 * (I[j, i] - M)**2 / VAR))
            #print(I[j, i], g[j, i])
    
    # Show Normazlied image
    img = Image.fromarray(g)
    img.show()
    #img.save('test.png')