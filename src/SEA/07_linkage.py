import os
import pickle 
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
np.set_printoptions(precision=3, suppress=True)


logfname = 'linkage_log.txt' 


def logging(msg, pt=True):
    if pt:
        print(msg)
    dt = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    with open(logfname,'a') as f:
        f.write(dt)
        f.write('\t')
        f.write(msg)
        f.write('\n')


def main():
    print('# Create log file')
    if not os.path.exists(logfname):
        f = open(logfname,'w') 
        f.close()
    
    logging(f'# Load pvalues')
    pvalues = np.load('pvalues.npy') # has nan
    
    logging('# Check symmetric and min/max values')
    logging(f'pvalues: {np.array_equal(pvalues, pvalues.T, equal_nan=True)}')
    logging(f"min pvalues: {np.nanmin(pvalues)} max pvalues: {np.nanmax(pvalues)}")
    logging(f'min pvalues: {np.nanmin(pvalues[np.where(pvalues>0)])} (except 0.0)')
    
    logging(f'# Make a diagonal as zero')
    pvalues = np.nan_to_num(pvalues, nan=0) # nan --> 0
    logging(f"min: {np.min(pvalues)} max: {np.max(pvalues)}")
    
    logging(f'# Average linkage') # UPGMA
    dist = pvalues
    condensed_dist = squareform(dist)
    Z_avr = linkage(condensed_dist, method='average')
    logging(f'Z shape: {Z_avr.shape}')
    
    logging(f'# Save linkage result(=z)')
    with open('z_avr.bin', 'wb') as wf:
        pickle.dump(Z_avr, wf)


if __name__ == '__main__':
    main()
