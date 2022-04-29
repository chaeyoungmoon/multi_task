import random
import os
import sys
import numpy as np
import torch 
import torch.nn.functional as F
import torch.utils.data as td
import logging
import itertools


def create_dataloader_list_kd(inputs, labels, task_indices, batch_size, mode: str, teacher_path=None) -> dict:
    logging.info(f'current mode: {mode}')
    
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    
    using_inputs = inputs[:]
    using_labels = labels[:][:,task_indices]
    num_tasks = len(task_indices)
    logging.info(f'input shape: {using_inputs.shape}, label shape: {using_labels.shape}')
    
    datasetlist_dict = construct_dataset_list(using_inputs, using_labels, mode=mode, teacher_path=teacher_path) 
    dataset_list = datasetlist_dict.get(mode)
    teacher_dataset_list = datasetlist_dict.get('teacher') # for making teacher logits

    dataloader_list = list()
    teacher_dataloader_list = list()
    
    for tidx in range(num_tasks):
        task_dataset = dataset_list[tidx]
        if task_dataset is None:
            task_dataloader = None
        else: 
            task_dataloader = td.DataLoader(dataset=task_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
        dataloader_list.append(task_dataloader)
        
        if teacher_dataset_list is not None:
            task_teacherset = teacher_dataset_list[tidx]
            task_teacherloader = td.DataLoader(dataset=task_teacherset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            teacher_dataloader_list.append(task_teacherloader)
        
    if len(teacher_dataloader_list) == 0:
        teacher_dataloader_list = None
    dataloaderlist_dict = {mode: dataloader_list, 'teacher': teacher_dataloader_list}    
    return dataloaderlist_dict


def construct_dataset_list(inputs, labels, mode: str, teacher_path=None) -> dict:     
    num_mols = labels.shape[0]
    num_tasks = labels.shape[1] 

    logging.info(f'num_mols: {num_mols}, num_tasks: {num_tasks}')
    
    total_train = 0
    task_dataset_list = []
    teacher_dataset_list = []
    for tidx in range(num_tasks): 
        
        task_dataset = SingleTaskDataset(inputs, labels[:,tidx], tidx, mode, teacher_path) 
        
        if len(task_dataset.using_indices) == 0:
            task_subset = None
        else:
            task_subset = td.Subset(dataset=task_dataset, indices=task_dataset.using_indices) # because of negative sampling
        task_dataset_list.append(task_subset)
        
        if task_dataset.teacher_indices is not None:
            teacher_subset = td.Subset(dataset=task_dataset, indices=task_dataset.teacher_indices)
            teacher_dataset_list.append(teacher_subset)
    
    if len(teacher_dataset_list) == 0:
        teacher_dataset_list = None
    task_dataset_dict = {mode: task_dataset_list, 'teacher': teacher_dataset_list}
    return task_dataset_dict


class SingleTaskDataset(td.Dataset):
    def __init__(self, ecfps, labels, tidx, mode, teacher_path=None):
        self.ecfps = ecfps
        self.labels = labels
        self.tidx = tidx
        self.mode = mode
        
        #### ----- using knowledge distillation ----- ####
        if not teacher_path is None:
            lname = f'teacher_logit.npy'
            teacher_logits = np.load(os.path.join(teacher_path, lname))
            
            ### make self.teacher_logits.shape[0] == self.labels.shape[0]
            self.teacher_logits = np.ones((labels.shape[0], 2)) * -1

        else:
            self.teacher_logits = None
        #### ----- using knowledge distillation ----- ####
        
        indices = np.arange(len(self.labels))
        self.pos_indices = indices[self.labels==1.0] 
        self.neg_indices = indices[self.labels==0.0]
        self.using_indices = indices[self.labels!=-1.0]
        self.teacher_indices = None
    
        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)
        self.num_use = len(self.using_indices)
    
        if mode == 'train':

            assert len(self.using_indices) == len(teacher_logits), \
            f"using_indicies {len(self.using_indices)} != teacher_logits {len(teacher_logits)}"
            
            self.teacher_logits[self.using_indices] = teacher_logits #### fill teacher
            self.using_indices = np.random.choice(self.using_indices, len(self.using_indices), replace=False) # shuffle   
            

    def __len__(self):
        return len(self.using_indices)

    
    def __getitem__(self, idx): 
        if not self.teacher_logits is None:
            ecfp = torch.from_numpy(self.ecfps[idx])
            label = np.eye(2)[self.labels[idx]] # one-hot vector
            label = torch.from_numpy(label) 
            teacher = torch.from_numpy(self.teacher_logits[idx])
            return {'ecfp': ecfp, 'label': label, 'teacher': teacher, 'task_idx': self.tidx} # ecfp & label: torch.tensor, task_idx: integer
        else:
            ecfp = torch.from_numpy(self.ecfps[idx])
            label = np.eye(2)[self.labels[idx]] # one-hot vector
            label = torch.from_numpy(label) 
            return {'ecfp': ecfp, 'label': label, 'task_idx': self.tidx} # ecfp & label: torch.tensor, task_idx: integer  

