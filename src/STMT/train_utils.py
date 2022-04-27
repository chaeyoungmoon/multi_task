import os
import random
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_and_log(message):
    logging.info(message)
#     print(message) 
    
    
def if_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
def load_dataset(config: dict):
    train_inputs = np.load(config['DATA']['TRAIN_INPUT'])
    train_inputs = np.float32(train_inputs)
    train_labels = np.load(config['DATA']['TRAIN_LABEL'])
    train_labels = np.int64(train_labels)
    
    val_inputs = np.load(config['DATA']['VAL_INPUT'])
    val_inputs = np.float32(val_inputs)
    val_labels = np.load(config['DATA']['VAL_LABEL'])
    val_labels = np.int64(val_labels)
    
    test_inputs = np.load(config['DATA']['TEST_INPUT'])
    test_inputs = np.float32(test_inputs)
    test_labels = np.load(config['DATA']['TEST_LABEL'])
    test_labels = np.int64(test_labels)
    
    print_and_log(f'train input shape: {train_inputs.shape}, train label shape: {train_labels.shape}')
    print_and_log(f'val input shape: {val_inputs.shape}, val label shape: {val_labels.shape}')
    print_and_log(f'test input shape: {test_inputs.shape}, test label shape: {test_labels.shape}')
    
    train = {'input': train_inputs, 'label': train_labels}
    val = {'input': val_inputs, 'label': val_labels}
    test = {'input': test_inputs, 'label': test_labels}
    
    return (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels)


def soft_cross_entropy_loss(logits, labels, temperature=1):
    '''
    cross entropy loss when using soft labels
    (labels is not an integer and sum is not 1)
    loss = - sum(label_0*logP_0 + label_1*logP_1)
    
    :logits: (batch_size, 2)
    :labels: (batch_size, 2)
    '''
    batch_size = logits.shape[0]
    log_probs = F.log_softmax(logits / temperature, dim=1)
    loss = - torch.sum(labels * log_probs) / batch_size
    return loss 