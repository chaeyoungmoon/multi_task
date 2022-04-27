import os
import sys
import argparse
import logging
import yaml
import pickle
from tqdm import tqdm
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset_stmt as ds
import net_mt as net
from stmt_trainer import Trainer
from train_utils import set_seed, print_and_log, if_mkdir, load_dataset


def cross_entropy_loss_for_val(logits, labels):
    ''' 
    logits.shape: (batch_size, 2)
    labels.shape: (batch_size,)
    cross entropy loss when label is not an integer 
    loss = - sum(label_0*logP_0 + label_1*logP_1)
    '''
    batch_size = logits.shape[0]
    labels = torch.stack([1-labels, labels], dim=1) # shape: (batch_size, 2)
    log_probs = F.log_softmax(logits, dim=1)
    loss = - torch.sum(labels * log_probs) / batch_size
    return loss 


def dump(data, path):
    '''
    dump data to path
    data should be dictionary type
    '''
    with open(path, 'w') as wf:
        yaml.dump(data, wf)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='../config.yaml', help='path for config file')
    parser.add_argument('-bs', '--batch-size', type=int, default=256, help='batch size (default: 256)')
    parser.add_argument('-ep', '--epochs', type=int, default=50, help='epochs (default: 50)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('-v', '--ver', default=None, help='version; (default: None)')
    parser.add_argument('--cluster', type=int, default=None, help='select a cluster (default: None)')
    parser.add_argument('-cf', '--cluster-first', type=int, default=None, help='select the first cluster (default: None)')
    parser.add_argument('-cl', '--cluster-last', type=int, default=None, help='select the last cluster (default: None)')
    parser.add_argument('--gpu', type=int, default=0, help='select a gpu (default: 0)')
    parser.add_argument('-h1', '--hidden1', type=int, default=1024, help='hidden1 nodes (default: 1024)')
    parser.add_argument('-h2', '--hidden2', type=int, default=128, help='hidden2 nodes (default: 128)')
    parser.add_argument('-dr', '--drop-rate', type=float, default=0.5, help='drop-out rate (default: 0.5)')
    parser.add_argument('-s', '--seed', type=int, default=102, help='random seed (default: 102)')
    parser.add_argument('--teacher', default=None, help='teacher [ST, MT] (default: None)')
    parser.add_argument('--patience', type=int, default=5, help='patience (default: 5)')
    
    args = parser.parse_args()
    print(f"{args.teacher}MT-{args.epochs}-{args.batch_size}-{args.learning_rate}-{args.drop_rate}-{args.seed}")

    assert args.teacher is not None, 'select the teacher (ST or MT)'
    
    with open(args.config) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    
    # set random seed 
    set_seed(args.seed)
    
    # set device 
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    print('Available devices: ', torch.cuda.device_count())
        
    # create a result and log directory
    if_mkdir('log')
    if_mkdir('results')
    if args.ver is None:
        version = f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.drop_rate}_{args.seed}'
    else:
        version = f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.drop_rate}_{args.seed}_{args.ver}'
    resdir = os.path.join('results', f'results{version}')
    logdir = os.path.join('log', f'log{version}') 
    if_mkdir(resdir)
    if_mkdir(logdir)
    
    # convert config into dictionary
    opt = vars(args).copy()
    config_path = os.path.join(resdir, f'current_config.yaml')
    opt.pop('cluster_first')
    opt.pop('cluster_last')
    opt.pop('cluster')
    dump(opt, config_path)
        
    # set the base log file
    logfname = os.path.join(logdir, f'{args.teacher}MT.base.train.log')
    logging.basicConfig(filename=logfname,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', 
                        force=True)
    print_and_log("===== start =====")
    print_and_log(f'epochs: {args.epochs} | batch-size: {args.batch_size} | learning-rate: {args.learning_rate}')
    print_and_log(f'hidden1: {args.hidden1} | hidden2: {args.hidden2} | drop-rate: {args.drop_rate}')
    print_and_log(f'temperature: {args.temperature} | alpha: {args.alpha}') # <------------- for KD
    print_and_log(f'result dir: {resdir}')

    # load common data   
    train_inputs = np.load(config['SEADATA']['TRAIN_INPUT'])
    train_inputs = np.float32(train_inputs)
    train_labels = np.load(config['SEADATA']['TRAIN_LABEL'])
    train_labels = np.int64(train_labels)
    
    val_inputs = np.load(config['SEADATA']['VAL_INPUT'])
    val_inputs = np.float32(val_inputs)
    val_labels = np.load(config['SEADATA']['VAL_LABEL'])
    val_labels = np.int64(val_labels)
    
    test_inputs = np.load(config['SEADATA']['TEST_INPUT'])
    test_inputs = np.float32(test_inputs)
    test_labels = np.load(config['SEADATA']['TEST_LABEL'])
    test_labels = np.int64(test_labels)
    
    print_and_log(f'train input shape: {train_inputs.shape}, train label shape: {train_labels.shape}')
    print_and_log(f'val input shape: {val_inputs.shape}, val label shape: {val_labels.shape}')
    print_and_log(f'test input shape: {test_inputs.shape}, test label shape: {test_labels.shape}')

    ## cluster dictionary: keys=tids, tinds, size
    with open(config['SEADATA']['CLUSTER'], 'rb') as f:
        cluster_tasks = pickle.load(f)
    
    # --------- base condition is identical regardless of clusters --------- #
    
    # train and test the model on every cluster
    if args.cluster_first is None:
        clstr1 = 0
    else:
        clstr1 = args.cluster_first
    if args.cluster_last is None:
        clstr2 = len(cluster_tasks)-1
    else:
        clstr2 = args.cluster_last
    print_and_log(f'cluster_first: {clstr1}, cluster_last: {clstr2}')
    
    for clstr in range(clstr1, clstr2+1):
        
        # make cluster directory
        cname = f'cluster{clstr:0>3}'    
        cluster_dir = os.path.join(resdir, cname)
        if_mkdir(cluster_dir)
        
        # get the tinds of cluster
        task_indices = cluster_tasks[clstr]['tinds']
        num_tasks = len(task_indices)
        
        # load best hyperparameters
        best_cluster_dir = os.path.join(f'../MT', 'results', f'best{args.seed}', cname) # --- since student is MT
        with open(os.path.join(best_cluster_dir, 'best_res_dir.yaml'), 'r') as rf:
            best_res_dir = yaml.load(rf, Loader=yaml.Loader)
        best_dir = best_res_dir['best_res_dir'] 
        best_dir = os.path.join('../MT/results', best_dir) # --- since student is MT
        best_config = os.path.join(best_dir, 'current_config.yaml')
        with open(best_config) as yml:
            best_opt = yaml.load(yml, Loader=yaml.FullLoader)
            
        # save hyperparameter into cluster### / current_config 
        opt = best_opt
        config_path = os.path.join(cluster_dir, f'current_config.yaml')
        opt['cluster'] = clstr
        opt['num_tasks'] = num_tasks
        dump(opt, config_path)
        
        
        # make hyperparameter dictionary
        hypers = {
            'batch_size': opt['batch_size'],
            'epochs': opt['epochs'],
            'learning_rate': opt['learning_rate'],
            'input_dim': 2048,
            'hidden1_dim': opt['hidden1'],
            'hidden2_dim': opt['hidden2'],
            'output_dim': 2,
            'device': device,
            'drop_rate': opt['drop_rate'],
            'patience': opt['patience'],
            'device': device,
            'cluster': opt['cluster'],
            'num_tasks': num_tasks
        }

        # make directory dictionary
        dirs = {
            'cluster_dir': cluster_dir,
            'task_dir': {}
        }
        
        
        # ===== knolwedge distillation: using teacher ===== #
        best_cluster_dir = os.path.join('results', f'best{args.seed}', cname)
        teacher_path = f'../{args.teacher}/{best_cluster_dir}'
        # ===== knolwedge distillation: using teacher ===== #

    
        # set the cluster specific log file
        logfname = os.path.join(logdir, f'{args.teacher}MT.c{clstr}.train.log')
        logging.basicConfig(filename=logfname,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', 
                            force=True)
        print_and_log("===== start =====")
        print_and_log(f"epochs: {hypers['epochs']} | batch-size: {hypers['batch_size']} | learning-rate: {hypers['learning_rate']}")
        print_and_log(f"hidden1: {hypers['hidden1_dim']} | hidden2: {hypers['hidden2_dim']} | drop-rate: {hypers['drop_rate']}")
        print_and_log(f'result dir: {best_dir}')
        print_and_log(f'current cluster: {clstr} | num tasks: {num_tasks}')

        # generate data loaders
        batch_size = hypers['batch_size']
        
        print_and_log('generate train dataloader')
        train_dataloader_dict = ds.create_dataloader_list_kd(
            train_inputs, train_labels, task_indices, batch_size, mode='train', teacher_path=teacher_path) 
        train_dataloader_list = train_dataloader_dict['train'] # length=1
        print_and_log(f'train dataloader #: {len(train_dataloader_list)}')
        
        print_and_log('generate val dataloader')
        val_dataloader_dict = ds.create_dataloader_list_kd(
            val_inputs, val_labels, task_indices, batch_size, mode='val')
        val_dataloader_list = val_dataloader_dict['val']
        print_and_log(f'val dataloader #: {len(val_dataloader_list)}')
        
        print_and_log('generate test dataloader')
        test_dataloader_dict = ds.create_dataloader_list_kd(
            test_inputs, test_labels, task_indices, batch_size, mode='test') 
        test_dataloader_list = test_dataloader_dict['test'] # length=num_tasks
        print_and_log(f'test dataloader #: {len(test_dataloader_list)}')

        # build model, train, save
        trainer = Trainer(hypers, dirs)
        trainer.build()

        print_and_log('train model and save prediction')
        trainer.train_and_predict(train_dataloader_list, val_dataloader_list, test_dataloader_list) # self.results
        


if __name__ == '__main__':
    main() 
