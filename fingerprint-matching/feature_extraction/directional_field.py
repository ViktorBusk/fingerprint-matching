from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from .normalize import normalize
from math import pi
import cv2 

class DirectionalField():
    def __init__(self, fingerprint: Image): 
        # 2D image array
        self.I = np.array(fingerprint) #248 x 338
        # Blocks
        self.W = 12 # Block size (W X W)
        self.blocks_x = self.I.shape[1] // self.W # W = 12, Width = 248 -> 20
        self.blocks_y = self.I.shape[0] // self.W # W = 12, Height = 338 -> 28
        # Field
        self.diff_x = self.I.shape[1] - self.blocks_x * self.W # W = 12 -> 8 (248 - 20 * 12)
        self.diff_y = self.I.shape[0] - self.blocks_y * self.W # W = 12 -> 2 (338 - 28 * 12)
        self.margin_left = self.diff_x // 2 # W = 12 -> 4 (px)
        self.margin_top = self.diff_y // 2 # W = 12 -> 1 (px)

        self.O = np.zeros([self.blocks_y, self.blocks_y], dtype = float)
        self.Vx = np.zeros([self.blocks_y, self.blocks_y], dtype = float)
        self.Vy = np.zeros([self.blocks_y, self.blocks_y], dtype = float)
        self.Phi_x = np.zeros([self.blocks_y, self.blocks_y], dtype = float)
        self.Phi_y = np.zeros([self.blocks_y, self.blocks_y], dtype = float)
        self.O_prime = np.zeros([self.blocks_y, self.blocks_y], dtype = float)

        # Gradient
        self.Gx, self.Gy = None, None
     
        #-----------------Get directionalField-----------------------
        self.I = normalize(self.I, 100, 100)
        self._calculate_gradient()
        self._calculate_local_block_orientation()
        self._low_pass_filter()

        self.plot()
        #img = Image.fromarray(self.I).show()
        #img.show()

    def _calculate_gradient(self):
        '''
        Compute gradients for each pixel using Sobel operator (used for edge detection).
        '''
        self.Gx = sobel(self.I, 0)  # horizontal derivative
        self.Gy = sobel(self.I, 1)  # vertical derivative

    def _calculate_local_block_orientation(self):
        '''
        Compute local orientation of the blocks using least square estimate.
        '''
        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                for u in range(i*self.W + self.margin_left, (i+1)*self.W + self.margin_left):
                    for v in range(j*self.W + self.margin_top, (j+1)*self.W + self.margin_top):
                        self.Vx[j, i] += (2*self.Gx[v, u]*self.Gy[v, u])
                        self.Vy[j, i] += (self.Gx[v, u]**2 - self.Gy[v, u]**2)
                        
                if self.Vx[j, i] != 0: # denominator can't be zero, arctan of 0 is 0
                    self.O[j, i] = (1/2) * np.arctan(self.Vy[j, i] / self.Vx[j, i]) # θ(i, j))

    def _low_pass_filter(self):
        # Remove loops later
        for i in range(self.blocks_x):
            for j in range(self.blocks_y):
                self.Phi_x[j, i] = np.cos(2*self.O[j, i]) #np.hypot(self.Vx[j, i], self.Vy[j, i]) * np.cos(2*self.O[j, i])
                self.Phi_y[j, i] = np.sin(2*self.O[j, i]) #np.hypot(self.Vx[j, i], self.Vy[j, i]) * np.sin(2*self.O[j, i])

        # map array from -1, 1 to 0 to 255
        self.Phi_x = np.uint8(255 * (self.Phi_x + 1) / 2) # WARNING: Looses precision on decimal if number is uneven i.e 255/2 = 127.5 -> 127
        self.Phi_y = np.uint8(255 * (self.Phi_y + 1) / 2) # WARNING: Looses precision on decimal if number is uneven i.e 255/2 = 127.5 -> 127

        # Applay noise reduction filter (low pass filter)
        self.Phi_x = cv2.medianBlur(self.Phi_x, 5)
        self.Phi_y = cv2.medianBlur(self.Phi_y, 5)

        # map bakc from 0, 255 to -1 to 1
        self.Phi_x = (self.Phi_x / 255) * 2 - 1
        self.Phi_y = (self.Phi_y / 255) * 2 - 1

        self.O_prime = (1/2)*np.arctan(self.Phi_y / self.Phi_x)
   
    def plot(self):
        fig, ax = plt.subplots()
        ax.imshow(self.I, cmap = 'gray')
        #ax.axis('off')
        for x in range(self.blocks_x):
            for y in range(self.blocks_y):
                i = x * self.W + self.W//2 + self.margin_left
                j = y * self.W + self.W//2 + self.margin_top
               
                X0 = i + self.W/2    
                Y0 = j + self.W/2
                r = self.W/2

                X1 = r*np.cos(self.O_prime[x, y]-pi/2)+X0
                Y1 = r*np.sin(self.O_prime[x, y]-pi/2)+Y0
                X2 = X0-r*np.cos(self.O_prime[x, y]-pi/2)
                Y2 = Y0-r*np.cos(self.O_prime[x, y]-pi/2)

                ax.quiver(X1, Y1, X2, Y2, color = 'blue')
        #plt.axes().set_aspect('equal')
        plt.show()
        np.pi


      
        