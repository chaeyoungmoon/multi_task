import pickle
import numpy as np
from scipy.cluster.hierarchy import fcluster


def main():
    with open('z_avr.bin', 'rb') as rf:
        Z = pickle.load(rf)
    print(np.min(Z[np.where(Z>0)]))
    
    thres=1e-50 # threshold pvalues
    f = fcluster(Z, t=thres, criterion='distance') 
    print(f"cut-off distance: {thres}")
    print(f"number of targets: {len(f)}") 
    print(f"number of clusters: {max(f)}")
    
    # cls_tinds {cluster: [tinds]}
    cls_tinds = {}
    for fi in range(len(f)):
        if not f[fi] in cls_tinds.keys():
            cls_tinds[f[fi]] = []
        cls_tinds[f[fi]].append(fi)
    print(f"cluster counts: {len(cls_tinds)}")
    
    # cluster_sizes {cluster no: cluster size}
    cluster_sizes = {}
    for c in cls_tinds.keys():
        cluster_sizes[c] = len(cls_tinds[c])

    # size_cnt: {cluster size: cluster counts}
    cluster_sizes_list = np.asarray(list(cluster_sizes.values()))
    uniques, counts = np.unique(cluster_sizes_list, return_counts=True)
    size_cnt = dict(zip(uniques, counts))
    print(size_cnt)
    
    # size_clusters {size: [cluster no.]}
    size_clusters = {}
    for c, s in cluster_sizes.items():
        if not s in size_clusters.keys():
            size_clusters[s] = []
        size_clusters[s].append(c)
        
    # sort cluster sizes (descending order)
    sizes = np.fromiter(size_clusters.keys(), dtype=int)
    sizes = np.sort(sizes)[::-1]
    print('cluster size', sizes)

    # line clusters up based on their size
    clusters = []
    for sz in sizes:
        clstrs = size_clusters[sz]
        clusters.extend(clstrs)
        
    # produce new cluster number
    to_new_keys_dict = dict(zip(clusters, range(len(clusters))))

    # assign new cluster number
    cluster_dict = {}
    cnt = 0
    for k in cls_tinds.keys():
        tinds = np.array(cls_tinds[k])
        tinds = np.sort(tinds)
        sz = len(tinds)
        if sz > 1:
            nk = to_new_keys_dict[k] # new key
            cluster_dict[nk] = {}
            cluster_dict[nk]['tinds'] = tinds
            cluster_dict[nk]['size'] = len(tinds)
            cnt += sz

    print(f"the number of clusters: {len(cls_tinds)} --> {len(cluster_dict)}")
    print(f"the number of targets: {len(f)} --> {cnt}")
    
    # save
    negative_log_cutoff = -np.log10(np.array([thres])).astype(int).item() # ex) 1e-50 --> 50
    with open(f'cluster_d{negative_log_cutoff}_dict.pickle', 'wb') as wf:
        pickle.dump(cluster_dict, wf)
        
        
if __name__ == '__main__':
    main()