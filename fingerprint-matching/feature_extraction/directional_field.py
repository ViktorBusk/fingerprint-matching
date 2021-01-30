from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# TODO: Create a new 2D numpy array which can be devided into blocks of size WxW

class DirectionalField():
    def __init__(self, fingerprint: Image): 
        self._np_array_2D = np.array(fingerprint) #248 x 338
        #self.block_array = self.split_image() # 3D numpy array of blocks
        self.W = self._np_array_2D.shape[1] / 20 # Block size
        self._G, self._Gx, self._Gy = None, None, None
    
    def _split_image(self, n_boxes):
        '''
        Divide the input fingerprint image into WxW sized blocks
        '''
        return np.array_split(self._np_array_2D, n_boxes)
        #doesn't work with colors

    def _calculate_gradient(self):
        '''
        Compute gradients for each pixel using Sobel operator (used for edge detection)
        '''
        self._Gx = ndimage.sobel(self._np_array_2D, 0)  # horizontal derivative
        self._Gy = ndimage.sobel(self._np_array_2D, 1)  # vertical derivative
        self._G = np.hypot(self._Gx, self._Gy)  # magnitude

    def _least_squares_estimate(self):
        pass

    def plot_gradient(self):
        if self._G is None: self._calculate_gradient() # Calculate gradient if it hasn't been 
        
        normalized_G = self._G # Copy gradient
        normalized_G *= 255.0 / np.max(normalized_G)  # normalize (Q&D)
        normalized_G = np.round(normalized_G).astype(int) # round and convert to integer
        
        # plotting
        plt.close("all")
        plt.figure()
        plt.suptitle("Gradient, and its components along each axis")
        ax = plt.subplot("131")
        ax.axis("off")
        ax.imshow(normalized_G, cmap='gray')
        ax.set_title("G (Normalized)")

        ax = plt.subplot("132")
        ax.axis("off")
        ax.imshow(self._Gx, cmap='gray')
        ax.set_title("Gx")

        ax = plt.subplot("133")
        ax.axis("off")
        ax.imshow(self._Gy, cmap='gray')
        ax.set_title("Gy")
        plt.show()

      
        