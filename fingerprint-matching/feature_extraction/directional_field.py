from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import math

class Block():


    pass

class DirectionalField():
    def __init__(self, fingerprint: Image): 
        # 2D image array
        self._np_array_2D = np.array(fingerprint) #248 x 338
        # Blocks
        self.W = 12 # Block size (W X W)
        self.blocks_x = self._np_array_2D.shape[1] // self.W # W = 12, Width = 248 -> 20
        self.blocks_y = self._np_array_2D.shape[0] // self.W # W = 12, Height = 338 -> 28
        # Field
        self.field_orientation = np.empty([self.blocks_y, self.blocks_y], dtype = float)
        self.diff_x = self._np_array_2D.shape[1] - self.blocks_x * self.W # W = 12 -> 8 (248 - 20 * 12)
        self.diff_y = self._np_array_2D.shape[0] - self.blocks_y * self.W # W = 12 -> 2 (338 - 28 * 12)
        self.margin_left = self.diff_x // 2 # W = 12 -> 4 (px)
        self.margin_top = self.diff_y // 2 # W = 12 -> 1 (px)
        # Gradient
        self._G, self._Gx, self._Gy = None, None, None

        # Get directionalField
        self._calculate_gradient()
        self._calculate_local_block_orientation()
        self._low_pass_filter()
        self.plot()


    def _calculate_gradient(self):
        '''
        Compute gradients for each pixel using Sobel operator (used for edge detection).
        '''
        self._Gx = sobel(self._np_array_2D, 0)  # horizontal derivative
        self._Gy = sobel(self._np_array_2D, 1)  # vertical derivative
        self._G = np.hypot(self._Gx, self._Gy)  # magnitude

    def _calculate_local_block_orientation(self):
        '''
        Compute local orientation of the blocks using least square estimate.
        '''
        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                tan_inverse_input = 0
                for u in range(i*self.W + self.margin_left, (i+1)*self.W + self.margin_left):
                    for v in range(j*self.W + self.margin_top, (j+1)*self.W + self.margin_top):
                        if self._Gx[v, u]**2 - self._Gy[v, u]**2 != 0: # denominator in expression can't be 0
                            tan_inverse_input += (2*self._Gx[v, u]*self._Gy[v, u] / (self._Gx[v, u]**2 - self._Gy[v, u]**2))
            
                self.field_orientation[j, i] = (1/2) * np.arctan(tan_inverse_input) # θ(i, j))

    def plot(self):
        fig, ax = plt.subplots()
        ax.imshow(self._np_array_2D, extent=[0, self.blocks_x, 0, self.blocks_y], cmap = 'gray')
        #ax.axis('off')
        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                ax.quiver(i, j, np.cos(math.pi/2 + 2 * self.field_orientation[j, i]), np.sin(math.pi/2 + 2 * self.field_orientation[j, i]), color = 'blue')
        #plt.axes().set_aspect('equal')
        plt.show()
        np.pi

    def _low_pass_filter(self):
        #self.field_orientation *= 2
        self.field_orientation = gaussian_filter(self.field_orientation, 2)
        #W_Fi = 5
        #for i in range(self.blocks_x):
            #for j in range(self.blocks_y):
                #self.field_orientation[j, i] = np.cos(2 * self.field_orientation[j, i])  # Φx(i, j)
                #self.field_orientation[j, i] = np.sin(2 * self.field_orientation[j, i])  # Φy(i, j)
                #self.field_orientation[j, i] =  gaussian_filter(self.field_orientation[j, i], 2)

    def plot_gradient(self):
        if self._G is None: self._calculate_gradient() # calculate gradient if it's uninitialized
        
        normalized_G = self._G # copy gradient
        normalized_G *= 255.0 / np.max(normalized_G)  # normalize (Q&D)
        normalized_G = np.round(normalized_G).astype(np.uint8) # round and convert to 8 bit integer
        
        # plotting
        plt.close("all")
        plt.figure()
        plt.suptitle("Gradient, and its components along each axis")
        ax = plt.subplot(131)
        ax.axis("off")
        ax.imshow(normalized_G, cmap='gray')
        ax.set_title("G (Normalized)")

        ax = plt.subplot(132)
        ax.axis("off")
        ax.imshow(self._Gx, cmap='gray')
        ax.set_title("Gx")

        ax = plt.subplot(133)
        ax.axis("off")
        ax.imshow(self._Gy, cmap='gray')
        ax.set_title("Gy")
        plt.show()

      
        