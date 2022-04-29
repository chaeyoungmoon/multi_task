import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)


def symm2d(matrix):
    isnan = np.isnan(matrix)
    nan_rows, nan_cols = np.where(isnan)
    nan_locs = zip(nan_rows, nan_cols)
    for r,c in nan_locs:
        matrix[r,c] = matrix[c,r]
    return matrix


def xfunc(z):
    return -np.exp(-z*np.pi/(np.sqrt(6)-0.577215665))


 # function for z<=28
def ztop_small(xz):  
    p = 1 - np.exp(xz)
    return p


# function for z>28
def ztop_big(xz):
    p = - xz - np.power(xz, 2)/2 - np.power(xz, 3)/6
    return p


def main():
    sort_target_scores = np.load('sort_target_scores.npy')
    sort_inds = np.load('sort_inds.npy')

    print(sort_target_scores.shape)
    print(sort_inds.shape)
    
    return_inds = np.argsort(sort_inds)
    target_scores = sort_target_scores[return_inds].copy()
    target_scores = target_scores[:,return_inds] 
    
    zscores = target_scores[:,:,2].copy()
    zscores = symm2d(zscores)
    np.save('zscores.npy', zscores)
    
    pvalues = np.where(zscores<=28, ztop_small(xfunc(zscores)), ztop_big(xfunc(zscores))) 
    pvalues_nan_to_zero = np.nan_to_num(pvalues) # nan -> 0.0
    
    print(f'max: {np.nanmax(pvalues):.3f} min: {np.nanmin(pvalues):.3e}')
    print(f'nonzero min: {np.nanmin(pvalues[np.nonzero(pvalues)]):.3e}')
    
    np.save('pvalues.npy', pvalues)
    

if __name__ == '__main__':
    main()