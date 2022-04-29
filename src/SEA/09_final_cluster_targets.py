import os
import pickle
import numpy as np


def gather_and_sort_tinds(cdict):
    ''' gather and sort all the tinds of cluster dictionary '''
    num_clusters = len(cdict)
    total_tinds = np.array([], dtype=int)
    for k in range(num_clusters):
        tinds = cdict[k]['tinds']
        total_tinds = np.append(total_tinds, tinds)
    total_tinds = np.sort(total_tinds)
    return total_tinds


def convert_tinds_to_tids(tinds, tind2tid_dict):
    tids = [tind2tid_dict[ti] for ti in tinds]
    return np.array(tids)


def main():
    dict_path = 'cluster_d50_dict.pickle'
    with open(dict_path, 'rb') as rf:
        cluster_dict = pickle.load(rf)
    print(cluster_dict[0].keys())
    
    dict_path = '../data23/from_scratch.v2/SEA/tid2tind.pickle'
    with open(dict_path, 'rb') as rf:
        tid2tind = pickle.load(rf)
    print(len(tid2tind))
    
    total_tinds = gather_and_sort_tinds(cluster_dict)
    print(len(total_tinds))
    
    tind2tid = dict(zip(tid2tind.values(), tid2tind.keys()))
    print(len(tind2tid))
    
    total_tids = convert_tinds_to_tids(total_tinds, tind2tid)
    print(len(total_tids))

    np.save('final_cluster_d50_targets.npy', total_tids)
    np.savetxt('final_cluster_d50_targets.txt', total_tids, fmt='%d')
    

if __name__ == '__main__':
    main() 
