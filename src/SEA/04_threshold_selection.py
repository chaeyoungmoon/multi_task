import numpy as np


def main():
    chi_list = np.load('chi_list.npy')
    edv_chi = chi_list[:,0]
    norm_chi = chi_list[:,1]
    thres = np.arange(100)*0.01 + 0.0
    
    # best-fit threshold (minimum chi-statistic)
    ti = np.where(edv_chi==np.min(edv_chi))[0][0] # threshold index
    print(ti, thres[ti])
    

if __name__ == '__main__':
    main()