import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def logging(msg, pt=True):
    if pt:
        print(msg)
    dt = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    with open('chi_statistic_log.txt','a') as f:
        f.write(dt)
        f.write('\t')
        f.write(msg)
        f.write('\n')

        
def func(x, m, n):
    return m * (x**n) # x: product set size, y: RS mean or std.dev.


def get_seeds(start=0, step=11, num=100):
    seeds=np.arange(0,num)*step+start
    return seeds


def main():
    
    print('Create log file')
    f = open('chi_statistic_log.txt','w')
    f.close()
    
    seeds = get_seeds()
    thres = np.arange(0,1.0,0.01)

    chi_list = [] # list of [evd_chi, norm_chi]
    mean_param = [] # list of RS mean model parameters
    std_param = [] # list of RS std.dev. model parameters
    edv_param = [] # list of EDV distribution parameters
    norm_param = [] # list of norm distribution parameters
    
    for ti in range(len(thres)): # threshold index
        rss = {} 

        logging(f'threshold={thres[ti]}')

        for seed in seeds:
            with open(f'rawscore_dicts/rawscore_dict{seed}.bin', 'rb') as f:
                rs_dict = pickle.load(f) 
            keys = list(rs_dict.keys()) 
            cnt = [len(rs_dict[k]) for k in rs_dict.keys()] 

            for ki, k in enumerate(keys): # key: product set size
                temp = cnt[ki]//100 # 2
                for i in range(temp): # 0,1
                    ii = i*100 +ti # threshold index in list
                    rs = rs_dict[k][ii] # value[ii]: raw score at the current thresholds
                    if k in rss.keys():
                        rss[k].append(rs)
                    else:
                        rss[k] = []
                        rss[k].append(rs)

        mean_dict = {}
        std_dict = {}
        for k in rss.keys():
            rss[k] = np.asarray(rss[k])
            mean_dict[k] = np.mean(rss[k]) # mean RS
            std_dict[k] = np.std(rss[k]) # std. dev.

        xdata = np.asarray(list(mean_dict.keys()))
        ydata = np.asarray(list(mean_dict.values()))
        popt, _ = curve_fit(func, xdata, ydata)
        exp_mean = func(np.unique(xdata), *popt)
        mean_param.append(popt)

        xdata = np.asarray(list(std_dict.keys()))
        ydata = np.asarray(list(std_dict.values()))
        popt, _ = curve_fit(func, xdata, ydata)
        exp_std = func(np.unique(xdata), *popt)
        std_param.append(popt)

        # set_size의 expected mean and std
        prod_size = np.unique(xdata)
        exp_val = {}
        for i in range(len(prod_size)):
            size = prod_size[i]
            exp_val[size] = (exp_mean[i], exp_std[i])

        # normalize raw score to z-score by using mean and std
        all_zs = np.array([])
        for k in rss.keys(): # key=product size
            zscores = (rss[k]-exp_val[k][0])/exp_val[k][1]
            all_zs = np.append(all_zs, zscores)

        # fit to EVD(gumbel distribution) and Normal
        loc, scale = stats.gumbel_r.fit(all_zs)
        loc_n, scale_n = stats.norm.fit(all_zs)
        edv_fitted = stats.gumbel_r(loc=loc, scale=scale)
        norm_fitted = stats.norm(loc=loc_n, scale=scale_n)
        edv_param.append([loc, scale])
        norm_param.append([loc_n, scale_n])

        # observed frequency
        bin_left = np.floor(np.min(all_zs)*100)/100
        bin_right = np.ceil(np.max(all_zs)*100)/100
        bin_width = 0.01
        bins = (bin_right - bin_left)/bin_width +1
        bin_edges = bin_left + np.arange(bins)*bin_width
        hist, bin_edges = np.histogram(all_zs, bins=bin_edges)
        obs_freq = hist
        total = np.sum(obs_freq)
        deg = len(obs_freq)-1 # number of bins

        # expected frquency: EDV
        prob_left = edv_fitted.cdf(bin_edges[:-1])
        prob_right = edv_fitted.cdf(bin_edges[1:])
        prob_bin = prob_right-prob_left
        edv_freq = prob_bin*total

        # expected frquency: Normal
        prob_left = norm_fitted.cdf(bin_edges[:-1])
        prob_right = norm_fitted.cdf(bin_edges[1:])
        prob_bin = prob_right-prob_left
        norm_freq = prob_bin*total

        # remove observed and expected freqs are both zeros (avoid zero-denominator)
        edv_nonzero = np.where(edv_freq)[0]
        norm_nonzero = np.where(norm_freq)[0]
        obs_nonzero = np.where(obs_freq)[0]
        edv_remain = np.sort(np.union1d(edv_nonzero, obs_nonzero))
        norm_remain = np.sort(np.union1d(norm_nonzero, obs_nonzero))

        eobs_freq = obs_freq[edv_remain]
        edv_freq = edv_freq[edv_remain]
        nobs_freq = obs_freq[norm_remain]
        norm_freq = norm_freq[norm_remain]

        # the normalized chi-square, or the chi-square per degree of freedom
        # chi-square test statistic
        # Sum[(Fo-Fe)^2 / Fo+Fe] (Fo: observed freq, Fe: expected freq)
        # normalized chi square --> divded by degree of freedom 
        edv_chi = np.sum((eobs_freq-edv_freq)**2 / (eobs_freq+edv_freq)) / deg
        norm_chi = np.sum((nobs_freq-norm_freq)**2 / (nobs_freq+norm_freq)) / deg
        chi_list.append([edv_chi, norm_chi])
        logging(f'\tEDV: {edv_chi:.2f} Norm: {norm_chi:.2f}')

    # save outputs
    chi_list = np.asarray(chi_list)
    np.save('chi_list.npy', chi_list)
    mean_param = np.asarray(mean_param)
    np.save('mean_param.npy', mean_param)
    std_param = np.asarray(std_param)
    np.save('std_param.npy', std_param)
    edv_param = np.asarray(edv_param)
    np.save('edv_param.npy', edv_param)
    norm_param = np.asarray(norm_param)
    np.save('norm_param.npy', norm_param)


if __name__ == '__main__':
    main()