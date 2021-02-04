from PIL import Image
from scipy.ndimage import sobel
import numpy as np
import matplotlib.pyplot as plt
from .normalize import normalize
from cv2 import medianBlur as median_blur
from math import pi

class OrientationField():
    def __init__(self, fingerprint: Image): 
        # 2D image array
        self.__fingerprint = fingerprint #248 x 338
        self.__I = np.array(self.__fingerprint) 
        # Blocks
        self.W = 12 # Block size (W X W)
        self.blocks_x = self.__I.shape[1] // self.W # W = 12, Width = 248 -> 20
        self.blocks_y = self.__I.shape[0] // self.W # W = 12, Height = 338 -> 28
        diff_x = self.__I.shape[1] - self.blocks_x * self.W # W = 12 -> 8 (248 - 20 * 12)
        diff_y = self.__I.shape[0] - self.blocks_y * self.W # W = 12 -> 2 (338 - 28 * 12)
        self.__margin_left = diff_x // 2 # W = 12 -> 4 (px)
        self.__margin_top = diff_y // 2 # W = 12 -> 1 (px)
        # Field
        self.__O, self.__O_prime = None, None
        self.__Vx, self.__Vy = None, None
        self.__Phi_x, self.__Phi_y = None, None
        # Gradient
        self.__G, self.__Gx, self.__Gy = None, None, None

    @property
    def orientation_field(self):
        if self.__O_prime is None: self.calculate()
        return self.__O_prime

    @property
    def gradient(self):
        if self.__G is None:
            self.__normalize()
            self.__calculate_gradient()
        return self.__G

    @property
    def gradient_x(self):
        if self.__G is None:
            self.__normalize()
            self.__calculate_gradient()
        return self.__Gx
    
    @property
    def gradient_y(self):
        if self.__G is None:
            self.__normalize()
            self.__calculate_gradient()
        return self.__Gy

    def _normalize(self):
        '''
        Normalize grayscale image before gradient computation
        '''
        self.__I = normalize(self.__I, 100, 100)

    def _calculate_gradient(self):
        '''
        Compute gradients for each pixel using Sobel operator (used for edge detection).
        '''
        self.__Gx = sobel(self.__I, 0)  # horizontal derivative
        self.__Gy = sobel(self.__I, 1)  # vertical derivative
        self.__G = np.hypot(self.__Gx, self.__Gy)

    def _calculate_local_block_orientation(self):
        '''
        Compute local orientation of the blocks using least square estimate.
        '''
        # Allocate memory for the arrays
        self.__O = np.zeros([self.blocks_y, self.blocks_y], dtype = np.float32)
        self.__Vx = np.zeros([self.blocks_y, self.blocks_y], dtype = np.float32)
        self.__Vy = np.zeros([self.blocks_y, self.blocks_y], dtype = np.float32)
        self.__Phi_x = np.zeros([self.blocks_y, self.blocks_y], dtype = np.float32)
        self.__Phi_y = np.zeros([self.blocks_y, self.blocks_y], dtype = np.float32)

        # Loop over each pixel in imgae (block wise)
        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                for u in range(i*self.W + self.__margin_left, (i+1)*self.W + self.__margin_left):
                    for v in range(j*self.W + self.__margin_top, (j+1)*self.W + self.__margin_top):
                        self.__Vx[j, i] += self.__Gx[v, u]**2 - self.__Gy[v, u]**2
                        self.__Vy[j, i] += 2*self.__Gx[v, u]*self.__Gy[v, u]
                        
                self.__O[j, i] = (1/2)*np.arctan2(self.__Vy[j, i], self.__Vx[j, i]) # θ(i, j))
                # Convert into a continuous vector field
                self.__Phi_x[j, i] = np.cos(2*self.__O[j, i])
                self.__Phi_y[j, i] = np.sin(2*self.__O[j, i])

    def _low_pass_filter(self):
        '''
        Applay noise reduction filter, removes salt and pepper like noise (low pass filter)
        '''
        # Allocate memory for array
        self.__O_prime = np.zeros([self.blocks_y, self.blocks_y], dtype = np.float32)

        # map array from -1, 1 to 0 to 255 (values are centered around 0, -pi/2 to pi/2)
        self.__Phi_x = np.uint8(255 * (self.__Phi_x + 1) / 2) # WARNING: Looses precision on decimal if number is uneven i.e 255/2 = 127.5 -> 127
        self.__Phi_y = np.uint8(255 * (self.__Phi_y + 1) / 2) # WARNING: Looses precision on decimal if number is uneven i.e 255/2 = 127.5 -> 127

        # Applay noise reduction filter
        self.__Phi_x = median_blur(self.__Phi_x, 5)
        self.__Phi_y = median_blur(self.__Phi_y, 5)

        # map back from 0, 255 to -1 to 1
        self.__Phi_x = np.float32((self.__Phi_x / 255) * 2 - 1)
        self.__Phi_y = np.float32((self.__Phi_y / 255) * 2 - 1)
        
        # Get filtered orientation field
        self.__O_prime = (1/2)*np.arctan2(self.__Phi_y, self.__Phi_x)

    def calculate(self):
        self._normalize()
        self._calculate_gradient()
        self._calculate_local_block_orientation()
        self._low_pass_filter()

    def show_gradient(self):
        if self.__G is None:
            self._normalize()
            self._calculate_gradient()
        Image.fromarray(self.__G).show()

    def show(self):
        if self.__O_prime is None: self.calculate() # Calculate field before drawing anything
        
        fig, ax = plt.subplots()
        ax.imshow(self.__fingerprint, cmap = 'gray')

        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                x = i * self.W + self.W//2 + self.__margin_left
                y = j * self.W + self.W//2 + self.__margin_top

                x_dir = np.cos(self.__O_prime[j, i])
                y_dir = np.sin(self.__O_prime[j, i])

                ax.quiver(x, y, x_dir, y_dir, color = 'blue', headwidth=1, headlength = 0)
        plt.show()


      
        