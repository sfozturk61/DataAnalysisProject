import numpy as np
from PIL import Image
import matplotlip.pyplot as plt

class LatticeImage():
    ''' Class containing all image information of a EMCCD acquired image of a atom lattice'''

    
    def __init__(self, name, N, M):
        ''' Initialize empyt LatticeImage object

        Parameters
        ----------
        name : A reference for this image.
        N : Dimension of optical lattice.
        M : Number of pixels per one side of lattice side.
        '''

        # Store dimensions as member variables.
        self.N = N
        self.M = M 

        # Initialize empty numpy array of corresponding length.
        self.image = np.zeros((N, N, M, M))

    def load_from_jpeg(self, jpeg_path):
        '''Load image data from jpeg file'''

        # Load image as greyscale image.
        raw_image = Image.open(jpeg_path).convert('L')

        # Check if image size matches dimension of LatticeImage.
        target_dimension = self.M * self.N 

        if not raw_image.size[0] == target_dimension and raw_image.size[1] == target_dimension:
            error_msg = f"Image dimensions {raw_image.size} does not fit target dimension of ({target_dimension}, {target_dimension})"
            raise Exception(error_msg)

        # Assign raw image to member variable
        self.raw_image = raw_image 

    def structure_image(self):
        """Load raw data into pre-strucured arra self.image"""

        # Retrieve dimensions for convenience.
        M = self.M
        N = self.N

        # Load data into (M*N) * (M8N) array
        image_array = np.array(self.raw_image)

        # Iterate over lattice sites and fill them with image data.
        for i in range(N):
            for j in range(N):
                self.image[i, j] = image_array[i*M:(i+1)*M, j*M:(j+1)*M]

    def show_raw_image(self):
        '''Show raw image'''
        self.raw_image.show()

    def plot_image(self):
        '''Plots the structure image'''

