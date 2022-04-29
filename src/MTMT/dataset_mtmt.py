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
    logging.info(f' current mode: {mode}')
    
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    
    ## get current cluster's dataset by task_indices
    num_tasks = len(task_indices)
    using_inputs = inputs[:]
    using_labels = labels[:][:,task_indices]

    logging.info(f' input shape: {using_inputs.shape}, label shape: {using_labels.shape}')
    
    ## get task_dataset_list from datasetlist_dictionary; if there is teacher, get teacher_dataset_list, too
    datasetlist_dict = construct_dataset_list(using_inputs, using_labels, mode=mode, teacher_path=teacher_path)
    task_dataset_list = datasetlist_dict.get(mode)
    teacher_dataset_list = datasetlist_dict.get('teacher')  ## return None if there is not teacher
    
    
    ## get dataloader list
    ## training mode: one dataloader uses merged multi-task dataset
    ## training mode: each task has its own teacher dataloader                 
    if mode == 'train': 
        dataloader_list = list() # length=1 (train) 
        teacher_dataloader_list = list() # lenght=num_tasks (being teacher)
        dataset = MultiTaskDataset(task_dataset_list)
        batchsampler = MultiTaskBatchSampler(task_dataset_list, batch_size, shuffle=shuffle)
        dataloader = td.DataLoader(dataset=dataset, batch_sampler=batchsampler, num_workers=2, pin_memory=True)
        dataloader_list.append(dataloader) 
        
        if teacher_dataset_list is not None:
            for tidx in range(num_tasks):
                task_teacherset = teacher_dataset_list[tidx]
                task_teacherloader = td.DataLoader(dataset=task_teacherset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                teacher_dataloader_list.append(task_teacherloader)
                
    ## test/val mode: each task has its own dataloader
    ## test/val mode: there is no teacher dataloader
    else:
        dataloader_list = list() # length=num_tasks (test/val)
        teacher_dataloader_list = list() # length=0 (no teacher)
        for tidx in range(num_tasks):
            task_dataset = task_dataset_list[tidx]
            if task_dataset is None:
                task_dataloader = None
            else:
                test_dataset = task_dataset
                task_dataloader = td.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
            dataloader_list.append(task_dataloader)
    
    if len(teacher_dataloader_list) == 0:
        teacher_dataloader_list = None
    dataloaderlist_dict = {mode: dataloader_list, 'teacher': teacher_dataloader_list}    
    
    return dataloaderlist_dict


def construct_dataset_list(inputs, labels, mode: str, teacher_path=None) -> dict:        
    num_mols = labels.shape[0]
    num_tasks = labels.shape[1] 

    logging.info(f'num_mols: {num_mols}, num_tasks: {num_tasks}')
    logging.info(f'\t(positive size, negative size) -> (class weight)') 

    
    total_train = 0
    task_dataset_list = []
    teacher_dataset_list = []
    for tidx in range(num_tasks): 
        
        task_dataset = SingleTaskDataset(inputs, labels[:,tidx], tidx, mode, teacher_path) 
        task_subset = td.Subset(dataset=task_dataset, indices=task_dataset.using_indices) 
        task_dataset_list.append(task_subset)
        
        ## if being teacher
        if task_dataset.teacher_indices is not None: ## teacher indices are same as using indices of a task
            teacher_subset = td.Subset(dataset=task_dataset, indices=task_dataset.teacher_indices)
            teacher_dataset_list.append(teacher_subset)
    
    if len(teacher_dataset_list) == 0:
        teacher_dataset_list = None
    task_dataset_dict = {mode: task_dataset_list, 'teacher': teacher_dataset_list}
    return task_dataset_dict


# MultiTaskDataset and MultiTaskBatchSampler are 
# from https://github.com/namisan/mt-dnn with some modifications
class MultiTaskDataset(td.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.taskidx2dataset = dict(zip(range(len(dataset_list)), dataset_list))

    def __len__(self):
        return sum(len(dataset) for dataset in self.dataset_list)

    def __getitem__(self, idx):
        task_idx, sample_idx = idx
        item = self.taskidx2dataset[task_idx][sample_idx] 
        return item    


class MultiTaskBatchSampler(td.BatchSampler):
    def __init__(self, dataset_list, batch_size, shuffle=True, drop_last=False):
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # generate shuffled index batches from each task
        multi_task_batch_idx_list = [] 
        for dataset in dataset_list:
            single_task_batch_idx = self._get_shuffled_index_batches(len(dataset), self.batch_size, self.shuffle) 
            multi_task_batch_idx_list.append(single_task_batch_idx)
        self.batch_idx_list = multi_task_batch_idx_list # length=num_tasks
    
    @staticmethod 
    def _get_shuffled_index_batches(dataset_len, batch_size, shuffle):
        index_batches = [range(i, min(i+batch_size, dataset_len)) for i in range(0, dataset_len, batch_size)] 
        if shuffle:
            random.shuffle(index_batches)
        return index_batches
    
    def __len__(self): # total number of batches
        return sum(len(single_task_batches) for single_task_batches in self.batch_idx_list)
    
    def __iter__(self):
        all_task_iters = [] # list of single task batch generator (len=total number of tasks)
        all_task_indices = [] # list for task selection (len=total number of batches)
        all_task_indices_sep = [] # (len=total_number of tasks)
        
        for single_task_batches in self.batch_idx_list:
            single_task_iters = iter(single_task_batches)
            all_task_iters.append(single_task_iters)
        
        for task_idx in range(len(self.batch_idx_list)):
            single_task_batch_idx = self.batch_idx_list[task_idx]
            task_indices = [task_idx] * len(single_task_batch_idx) # task_idx * the number of batches in single task
            all_task_indices.extend(task_indices)
            all_task_indices_sep.append(task_indices)
        
        if self.shuffle:
            random.shuffle(all_task_indices)
        
        for task_idx in all_task_indices:
            batch = next(all_task_iters[task_idx])
            
            if not self.drop_last:
                yield [(task_idx, sample_idx) for sample_idx in batch]
            else:
                if len(batch) == self.batch_size:
                    yield [(task_idx, sample_idx) for sample_idx in batch] 
                

class SingleTaskDataset(td.Dataset):
    def __init__(self, ecfps, labels, tidx, mode, teacher_path=None, best_epoch=None):
        self.ecfps = ecfps
        self.labels = labels
        self.tidx = tidx
        self.mode = mode
        
        #### using teacher in training ####
        if not teacher_path is None:
            dname = f"task{tidx:0>2}"
            task_dir = os.path.join(teacher_path, dname)

            lname = f'teacher_logit.npy'
            teacher_logits = np.load(os.path.join(task_dir, lname))
            self.teacher_logits = np.ones((labels.shape[0], 2)) * -1 ## background for labels from teacher; fill in later

        else:
            self.teacher_logits = None
        #### using teacher training ####
        
        indices = np.arange(len(self.labels))
        self.pos_indices = indices[self.labels==1.0] 
        self.neg_indices = indices[self.labels==0.0]
        self.using_indices = indices[self.labels!=-1.0]
        self.teacher_indices = None
    
        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)
        self.num_use = len(self.using_indices)

        if mode == 'train':
            if self.teacher_logits is not None:
                assert len(self.using_indices) == len(teacher_logits), \
                f"using_indicies {len(self.using_indices)} != teacher_logits {len(teacher_logits)}"
                self.teacher_logits[self.using_indices] = teacher_logits # fill labels from teacher with teacher logits 
            
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


