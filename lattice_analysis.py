import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt

class AnalyzeLatticeImages():

    ''' Class analyzing generated images with different models.'''

    def __init__(self, N, M, std, x_loc, y_loc):
        ''' Initialize empty object

        Parameters
        ----------

        '''
        
        # Store dimensions as member variables.
        self.N = N
        self.M = M 
        self.std = std
        
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
        