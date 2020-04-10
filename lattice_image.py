import numpy as np
from PIL import Image

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

        # Load image.
        raw_image = Image.open(jpeg_path)

        # Check if image size matches dimension of LatticeImage.
        target_dimension = self.M * self.N 

        if not raw_image.size[0] == target_dimension and raw_image.size[1] == target_dimension:
            error_msg = f"Image dimensions {raw_image.size} does not fit target dimension of ({target_dimension}, {target_dimension})"
            raise Exception(error_msg)

        # Assign raw image to member variable
        self.raw_image = raw_image

    def show_raw_image():
        '''Show raw image'''

        self.raw_image.show()

