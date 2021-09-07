from PIL import Image
from scipy.ndimage import sobel
import numpy as np
import matplotlib.pyplot as plt
from .normalize import normalize
from cv2 import medianBlur as median_blur

class OrientationField():
    def __init__(self, fingerprint: Image, block_size: int): 
        # 2D image array
        self._fingerprint = fingerprint 
        self._I = np.array(self._fingerprint) 
        # Blocks
        self.W = block_size 
        self.blocks_x = self._I.shape[1] // self.W 
        self.blocks_y = self._I.shape[0] // self.W 
        # Margin
        self._margin_left, self._margin_top = self._get_margin()
        # Field
        self._O_prime = None
        self._Phi_x, self._Phi_y = None, None
        # Gradient
        self._G, self._Gx, self._Gy = None, None, None

    @property
    def orientation_field(self):
        if self._O_prime is None: 
            self.calculate()
        return self._O_prime

    @property
    def gradient_x(self):
        if self._Gx is None:
            self._normalize()
            self._calculate_gradient()
        return self._Gx
    
    @property
    def gradient_y(self):
        if self._Gy is None:
            self._normalize()
            self._calculate_gradient()
        return self._Gy

    @property
    def gradient(self):
        if self._G is None:
            self._normalize()
            self._calculate_gradient()
        return self._G

    def _get_margin(self):
        diff_x = self._I.shape[1] - self.blocks_x * self.W 
        diff_y = self._I.shape[0] - self.blocks_y * self.W 
        margin_left = diff_x // 2
        margin_top = diff_y // 2
        return margin_left, margin_top

    def _normalize(self):
        '''
        Normalize grayscale image before gradient computation
        '''
        self._I = normalize(self._I, 100, 100)

    def _calculate_gradient(self):
        '''
        Compute gradients for each pixel using Sobel operator (used for edge detection).
        '''
        self._Gx = sobel(self._I, 0)  # horizontal derivative
        self._Gy = sobel(self._I, 1)  # vertical derivative
        self._G = np.hypot(self._Gx, self._Gy)

    def _calculate_local_block_orientation(self):
        '''
        Compute local orientation of the blocks using least square estimate.
        '''
        # Initialize arrays 
        Vx = np.zeros([self.blocks_y, self.blocks_x], dtype = np.float32)
        Vy = np.zeros([self.blocks_y, self.blocks_x], dtype = np.float32)
        O = np.zeros([self.blocks_y, self.blocks_x], dtype = np.float32)
        self._Phi_x = np.zeros([self.blocks_y, self.blocks_x], dtype = np.float32)
        self._Phi_y = np.zeros([self.blocks_y, self.blocks_x], dtype = np.float32)

        # Loop over each pixel in imgae (block wise)
        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                for u in range(i*self.W + self._margin_left, (i+1)*self.W + self._margin_left):
                    for v in range(j*self.W + self._margin_top, (j+1)*self.W + self._margin_top):
                        Vx[j, i] += self.gradient_x[v, u]**2 - self.gradient_y[v, u]**2
                        Vy[j, i] += 2 * self.gradient_x[v, u] * self.gradient_y[v, u]
                        
                # OrientationField
                O[j, i] = (1/2)*np.arctan2(Vy[j, i], Vx[j, i]) # θ(i, j))

                # Convert into a continuous vector field
                self._Phi_x[j, i] = np.cos(2*O[j, i])
                self._Phi_y[j, i] = np.sin(2*O[j, i])

    def _low_pass_filter(self):
        '''
        Applay noise reduction filter, removes salt and pepper like noise (low pass filter)
        '''
        # map array from -1, 1 to 0 to 255 (values are centered around 0, -pi/2 to pi/2)
        self._Phi_x = np.uint8(255 * (self._Phi_x + 1) / 2) # WARNING: Looses precision on decimal if number is uneven i.e 255/2 = 127.5 -> 127
        self._Phi_y = np.uint8(255 * (self._Phi_y + 1) / 2) # WARNING: Looses precision on decimal if number is uneven i.e 255/2 = 127.5 -> 127

        # Applay noise reduction filter
        self._Phi_x = median_blur(self._Phi_x, 5)
        self._Phi_y = median_blur(self._Phi_y, 5)

        # map back from 0, 255 to -1 to 1
        self._Phi_x = np.float32((self._Phi_x / 255) * 2 - 1)
        self._Phi_y = np.float32((self._Phi_y / 255) * 2 - 1)
        
        # Get filtered orientation field
        self._O_prime = (1/2)*np.arctan2(self._Phi_y, self._Phi_x)

    def calculate(self):
        self._normalize()
        self._calculate_gradient()
        self._calculate_local_block_orientation()
        self._low_pass_filter()

    def show_gradient(self):
        Image.fromarray(self.gradient).show()

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self._fingerprint, cmap = 'gray')

        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                x = i * self.W + self.W//2 + self._margin_left
                y = j * self.W + self.W//2 + self._margin_top

                x_dir = np.cos(self.orientation_field[j, i])
                y_dir = np.sin(self.orientation_field[j, i])

                ax.quiver(x, y, x_dir, y_dir, color = 'blue' , headwidth=1, headlength = 0, scale = 40)
        plt.show()
