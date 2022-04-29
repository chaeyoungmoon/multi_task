import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs



def get_sim_list(set1, set2):
    sim_list = np.array([])
    for i in range(len(set1)): 
        query_fp = set1[i] # query = single fp of set1
        target_fps = set2 # targets = all fps of set2
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, target_fps)
        sim_list = np.append(sim_list, sims)
    return sim_list


def func(x, m, n):
    return m*x**n


def logging(msg, pt=True):
    if pt:
        print(msg)
    dt = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    with open('target_zscore_log.txt','a') as f: 
        f.write(dt)
        f.write('\t')
        f.write(msg)
        f.write('\n')


def main():
    
    THRESHOLD = 0.74 
    print(f'===== threshold: {THRESHOLD} =====')
    
    print('# Create log file')
    f = open('target_zscore_log.txt','w') 
    f.close()
    
    logging('# Start')
    logging('# Load data')
    ecfp = np.load('../data23/from_scratch.v2/SEA/tv_ecfp.npy')
    with open('tid_lids.bin', 'rb') as f:
        tid_lids = pickle.load(f)
    set_sizes = [len(tid_lids[tid]) for tid in tid_lids.keys()]
    mean_param_ = np.load('mean_param.npy') # (100,2)
    std_param_ = np.load('std_param.npy')
    num_tars = len(tid_lids.keys())
    logging(f'\tecfp: {len(ecfp)}') # ligand indices are based on raw ecfp
    logging(f'\tnum_tars: {num_tars}')
    logging(f'\tmax size: {max(set_sizes)}')
        
    logging('# Convert array to fingerprint')
    fps = []
    total = len(ecfp)
    for i, arr in enumerate(ecfp):
        bs = "".join(arr.astype(str))
        fp = DataStructs.cDataStructs.CreateFromBitString(bs)
        fps.append(fp)
    logging(f'\tfps: {len(fps)}')
    del ecfp
    
    logging('# Create ligand set and sort')
    lig_set = []
    for tid in tid_lids.keys():
        lids = tid_lids[tid]
        tar_fps = [fps[li] for li in lids]
        lig_set.append(tar_fps)
    sizes = np.array([len(s) for s in lig_set])
    sort_inds = np.argsort(sizes)
    sort_lig_set = [lig_set[i] for i in sort_inds]
    np.save('sort_inds.npy', sort_inds)
    
    thres = THRESHOLD 
    thresholds = np.arange(0,100)*0.01 + 0
    thind = np.where(thresholds==thres)[0][0] 
    logging(f'# Set threshold: {thres} threshold index: {thind}')
    mean_param = mean_param_[thind]
    std_param = std_param_[thind]

    logging('# Produce z-scores array...') 
    setlist_a = sort_lig_set.copy()
    setlist_b = sort_lig_set.copy()
    
    N = num_tars 

    container = np.ones((N,N,3)) 
    container[:] = np.NaN

    logging(f'container: {container.shape} :: size, rawscore, zscore')
    logging(f'threshold: {thres}')
    cnt = 0
    for i,seta in enumerate(setlist_a[:N]): 
        logging(f'\t{i}th set', pt=False)
        for j,setb in enumerate(setlist_b[:N]): 
            if i < j: 
                # size
                size = len(seta)*len(setb)
                container[i,j,0] = size
                
                # raw score
                sim_list = get_sim_list(seta, setb)
                sim_list = np.where(sim_list<thres, 0.0, sim_list) # if RS<threshold, RS=0
                rs = np.sum(sim_list)
                container[i,j,1] = rs

                # z-score
                exp_mean = func(size, *mean_param)
                exp_std = func(size, *std_param)
                zscore = (rs - exp_mean) / exp_std
                container[i,j,2] = zscore
                cnt += 1
                
            else:
                continue

    logging('...done')
    logging(f'\ttotal pairs: {cnt}')

    logging('# Save the array...')
    np.save('sort_target_scores.npy', container) 
    
    logging('# Finished')


if __name__ == '__main__':
    main()
