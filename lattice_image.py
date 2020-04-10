

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