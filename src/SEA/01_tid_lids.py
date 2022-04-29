import numpy as np
import pickle

np.random.seed(100)

def main():
    labels = np.load('../data23/from_scratch.v2/SEA/tv_labels.npy') # -1,0,1
    num_mols, num_tars = labels.shape
    print(f'num_mols: {num_mols} num_tars: {num_tars}')

    tid_lids_orig = {} # key: target index, values: ligand indices of target
    for tid in range(num_tars):
        lids = np.where(labels[:,tid]==1)[0]
        tid_lids_orig[tid] = lids

    tid_lids = {}
    for ti in tid_lids_orig.keys():
        lids = tid_lids_orig[ti]
        if len(lids) > 3000:
            chosen = np.random.choice(lids, 3000)
            tid_lids[ti] = chosen
        else:
            tid_lids[ti] = lids

    total_lids = np.array([])
    for ti in tid_lids.keys():
        lids = tid_lids[ti]
        total_lids = np.append(total_lids, lids)
        total_lids = np.unique(total_lids)
    total_lids = np.sort(total_lids)
    total_lids = total_lids.astype('int64')

    print(f'before: {num_mols} --> after: {total_lids.shape[0]}') # 478,395 -> 382,373
    print(f'data type: {total_lids.dtype}') 

    np.save('total_lids.npy', total_lids) # total_lids.npy == lids.npy
    with open('tid_lids.bin', 'wb') as f:
        pickle.dump(tid_lids, f)

if __name__ == '__main__':
    main() 