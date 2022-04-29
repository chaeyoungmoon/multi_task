import os
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


def logging(msg, pt=True):
    if pt:
        print(msg)
    dt = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    with open('raw_score_list_log.txt','a') as f:
        f.write(dt)
        f.write('\t')
        f.write(msg)
        f.write('\n')


def get_seeds(start, step, num):
    return np.arange(0,num)*step+start


def main():
    
    print('# Create log file')
    f = open('raw_score_list_log.txt','w')
    f.close()
    os.mkdir('./rawscore_dicts')
    
    logging('# Start')
    logging('# Load data')
    ecfp = np.load('../data23/from_scratch.v2/SEA/tv_ecfp.npy')
    raw = len(ecfp)
    lids = np.load('total_lids.npy')
    ecfp = ecfp[lids]
    logging(f'\tecfp: {raw} --> {len(ecfp)}')
    
    logging('# Convert array to fingerprint')
    fps = []
    total = len(ecfp)
    for i, arr in enumerate(ecfp):
        bs = "".join(arr.astype(str))
        fp = DataStructs.cDataStructs.CreateFromBitString(bs)
        fps.append(fp)
    logging(f'\tfps: {len(fps)}')
    
    logging('# Prepare edges: 10-1000')
    edges = np.arange(10,1001,10)
    logging(f'\tedges: {len(edges)}')

    logging('# Prepare thresholds: 0.0-0.99')
    thres = np.arange(0.0,1.0,0.01)
    logging(f'\tthres: {len(thres)}')
    
    seeds = get_seeds(0, 11, 100)
    for seed in seeds:
        np.random.seed(seed)
        logging(f'# Seed: {seed})')

        logging('# Prepare the background dataset')
        inds = np.arange(len(fps))
        bginds = np.random.choice(inds, size=2000, replace=False) 
        bgfpsa = [fps[bgi] for bgi in bginds[:1000]]
        bgfpsb = [fps[bgi] for bgi in bginds[1000:]]
        logging(f'\tbgfps: {len(bgfpsa)}, {len(bgfpsb)}')

        logging('# Produce rawscore_dict...') # key=size, value=[(rs, mean, std) for ts]
        nabs = []
        rawscore_dict = {}
        for i, a in enumerate(edges):
            logging(f'\tsize a: {a}', pt=False)
            seta = bgfpsa[:a]
            sizea = len(seta)
            for j, b in enumerate(edges[i+1:]): # A<B   
                setb = bgfpsb[:b]
                sizeb = len(setb)
                nab = sizea*sizeb # number of pairs
                nabs.append(nab)
                if not nab in rawscore_dict.keys():
                    rawscore_dict[nab] = []
                sim_list = get_sim_list(seta, setb)
                for thr in thres:
                    sim_list = np.where(sim_list>=thr, sim_list, 0.0) # lignad-ligand raw scores
                    rawscore = np.sum(sim_list) # set-set raw score
                    rawscore_dict[nab].append(rawscore)
        logging('...done')
        logging(f'\ttotal pairs: {len(nabs)}')

        logging('# Save the dictionary...')
        with open(f'rawscore_dicts/rawscore_dict{seed}.bin','wb') as f:
            pickle.dump(rawscore_dict, f)
        logging('# ...done')
    
    logging('# Finished')
    
if __name__ == '__main__':
    main()
