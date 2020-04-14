import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt

class AnalyzeLatticeImages():

    ''' Class containing all image information of an EMCCD acquired image of a atom lattice'''

    def __init__(self, N, M, std, x_loc, y_loc):
        ''' Initialize empty LatticeImage object

        Parameters
        ----------
        name : A reference for this image.
        N : Dimension of optical lattice.
        M : Number of pixels per one side of lattice side.
        jpeg_path: The path of the image file to be loaded.
        '''
        
        # Store dimensions as member variables.
        self.N = N
        self.M = M 
        self.std = std   
        
        # Store output results
        self.x_loc = x_loc
        self.y_loc = y_loc

    def mixture_model(self, x, y, std, xsite, ysite):

        with pm.Model() as test_model:

            #Prior
            P = pm.Uniform('P', lower=0, upper=1)

            xc = (xsite[0]+xsite[1])/2 #x center of the site
            yc = (ysite[0]+ysite[1])/2 #y center of the site

            #Photons scattered from the atoms are Gaussian distributed
            atom_x = pm.Normal.dist(mu=xc, sigma=std).logp(x)
            atom_y = pm.Normal.dist(mu=yc, sigma=std).logp(y)
            atom = atom_x + atom_y

            #Photons from the camera background are uniform distributed
            background_x = pm.Uniform.dist(lower = xsite[0], upper = xsite[1]).logp(x)
            background_y = pm.Uniform.dist(lower = ysite[0], upper = ysite[1]).logp(y)
            background = background_x + background_y

            #Log-likelihood
            log_like = tt.log((P * tt.exp(atom) + (1-P) * tt.exp(background)))

            pm.Potential('logp', log_like.sum())

        map_estimate = pm.find_MAP(model=test_model)
        P_value = map_estimate["P"][0]
        
        self.P_value = P_value
        
    def run_mixture_model(self, x_loc, y_loc, N, M, std):
        
        P_array = np.zeros((N,N))
        lims = np.arange(0, (N+1)*M, M) - (N*M)/2
        for ny in range(N):
            for nx in range(N):
                x = np.where((x_loc > lims[nx]) & (x_loc <= lims[nx+1]) & (y_loc > lims[-(ny+2)]) & (y_loc <= lims[-(ny+1)]), x_loc, np.pi)
                x_new = x[x != np.pi]
                y = np.where((x_loc > lims[nx]) & (x_loc <= lims[nx+1]) & (y_loc > lims[-(ny+2)]) & (y_loc <= lims[-(ny+1)]), y_loc, np.pi)
                y_new = y[y != np.pi]

                xsite = np.array([lims[nx], lims[nx+1]])
                ysite = np.array([lims[-(ny+2)], lims[-(ny+1)]])

                P_array[ny,nx] = mixture_model(x_new, y_new, std, xsite, ysite)
                
        self.P_array = P_array