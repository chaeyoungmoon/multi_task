import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as special
from sklearn import metrics
import net_mt as net
from train_utils import print_and_log, if_mkdir, soft_cross_entropy_loss


class Trainer():
    def __init__(self, hypers, dirs):
        super().__init__()
        self.hypers = hypers
        self.model = None 
        self.dirs = dirs
    
    def check_model(self):
        if self.model is None:
            print_and_log("build a model first")
            sys.exit()
    
    def build_metric_lists(self):
        num_tasks = self.hypers['num_tasks']
        
        ## train/val/test loss & auc
        train_loss_list_dict = {} # len=num_tasks
        val_loss_list_dict = {} # len=num_tasks
        test_loss_dict = {}
        train_auc_list_dict = {}
        val_auc_list_dict = {}
        test_auc_dict = {}
        
        for tidx in range(num_tasks):
            train_loss_list_dict[tidx] = [] # len=num_epochs
            val_loss_list_dict[tidx] = [] # len=num_epochs
            test_loss_dict[tidx] = None
            train_auc_list_dict[tidx] = []
            val_auc_list_dict[tidx] = []
            test_auc_dict[tidx] = None
            
        self.train_loss_list_dict = train_loss_list_dict
        self.val_loss_list_dict = val_loss_list_dict
        self.test_loss_dict = test_loss_dict
        self.train_auc_list_dict = train_auc_list_dict
        self.val_auc_list_dict = val_auc_list_dict
        self.test_auc_dict = test_auc_dict
        
        self.val_loss_list_dict['min'] = [None] * num_tasks ## container for min loss

        task_stop_bool = np.zeros(num_tasks) # len=num_tasks 
        task_stop_bool = task_stop_bool.astype(bool)
        self.task_stop_bool = task_stop_bool
        self.task_stop_dict = {} ## save stop_task_index
        self.task_patiences = np.ones(num_tasks) * -1
    
    
    def build(self):
        kwargs = self.hypers
        
        num_tasks = self.hypers['num_tasks']
        lr = self.hypers['learning_rate']
        num_epochs = self.hypers['epochs']
        device = self.hypers['device']

        
        self.model = net.Net(**kwargs).to(device)
        self.model_kwargs = kwargs
        
        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.train_loss_fn = soft_cross_entropy_loss
        self.val_loss_fn = nn.CrossEntropyLoss()
        
        
        shared_layers = self.model.shared_layers
        tasks_layers = self.model.task_specific_layers
        
        self.shared_optimizer = torch.optim.Adam(shared_layers.parameters(), lr=lr)
        self.shared_optimizer.zero_grad()
        self.task_optimizer = torch.optim.Adam(tasks_layers.parameters(), lr=lr) ###
        self.task_optimizer.zero_grad()
        self.task_param_list = list(self.model.task_specific_layers.named_parameters()) # [0.weights, 0.bias, 1.weights, 1.bias, ...]
        
        self.build_metric_lists()


    def freeze_except_training(self, task_idx):
        task_weight_idx = task_idx*2
        task_bias_idx = task_idx*2+1
        for param_idx, named_param in enumerate(self.task_param_list):
            name, param = named_param
            if param_idx == task_weight_idx or param_idx == task_bias_idx:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    
    def train_single_epoch(self, train_loaders, current_epoch):
        self.check_model()
        model = self.model
        model.train()
        device = self.hypers['device']
        num_tasks = self.hypers['num_tasks']
        num_epochs = self.hypers['epochs']
        
        true_rate = current_epoch/num_epochs
        if current_epoch % 20 == 0:
            print_and_log(f'\t current rate: {true_rate:.4f} at {current_epoch}/{num_epochs}')
        
        student_loss_fn = self.student_loss_fn
        distill_loss_fn = self.distill_loss_fn
        loss_fn = self.train_loss_fn
        
        shared_optimizer = self.shared_optimizer
        task_optimizer = self.task_optimizer
        task_param_list = self.task_param_list
        
        train_loader = train_loaders[0] # len(train_loaders)=1
        running_loss = dict()
        logits_dict = {}
        labels_dict = {}
        for tidx in range(num_tasks):
            running_loss[tidx] = []
            logits_dict[tidx] = np.empty((0,2), float)
            labels_dict[tidx] = np.empty((0,2), float)
        
        for step, batch in enumerate(train_loader):
            inputs_batch = batch['ecfp'].to(device, non_blocking=True)
            true_labels_batch = batch['label'].to(device, non_blocking=True)
            teacher_logits_batch = batch['teacher'].to(device, non_blocking=True)
            teacher_labels_batch = F.softmax(teacher_logits_batch, dim=1)
            
            task_idx_batch = batch['task_idx'].to(device, non_blocking=True)
            task_idx = task_idx_batch[0].detach().item()
            
            if self.task_stop_bool[task_idx]:
                continue
            
            labels_batch = (true_rate*true_labels_batch 
                            + (1-true_rate)*teacher_labels_batch)
            
            self.freeze_except_training(task_idx)

            outputs_batch = model(inputs_batch, task_idx)
            
            loss_batch = loss_fn(outputs_batch, labels_batch)
            running_loss[task_idx].append(loss_batch.item())
            
            ## save logits
            outputs_batch = outputs_batch.detach().cpu().numpy()
            logits_dict[task_idx] = np.append(logits_dict[task_idx], outputs_batch, axis=0)
            
            ## save labels
            true_labels_batch = true_labels_batch.detach().cpu().numpy()
            labels_dict[task_idx] = np.append(labels_dict[task_idx], true_labels_batch, axis=0)

            shared_optimizer.zero_grad()
            task_optimizer.zero_grad()
            loss_batch.backward()
            shared_optimizer.step()
            task_optimizer.step()

        for tidx in range(num_tasks):
            if self.task_stop_bool[tidx]: # if stop, running_loss[tidx] is empty
                continue
                
            ## hold epoch loss
            epoch_loss = np.mean(np.array(running_loss[tidx]))
            self.train_loss_list_dict[tidx].append(epoch_loss)
            
            ## hold epoch auc
            logits = logits_dict[tidx]
            labels = labels_dict[tidx]
            epoch_auc = self.calculate_auc(logits, labels)
            self.train_auc_list_dict[tidx].append(epoch_auc)


    def val_single_epoch(self, val_loaders, epoch):
        self.check_model()
        num_tasks = self.hypers['num_tasks']
        
        test_results = dict()
        for tidx in range(num_tasks):
            val_loader = val_loaders[tidx]
            if self.task_stop_bool[tidx]:
                continue
            self.val_single_model_single_epoch(val_loader, tidx, epoch)

            
    def val_single_model_single_epoch(self, val_loader, tidx, current_epoch):
        self.check_model()
        model = self.model
        model.eval()
        loss_fn = self.val_loss_fn
        device = self.hypers['device']
        num_tasks = self.hypers['num_tasks']
        
        # create the test result container
        logits = np.empty((0,2), float)
        labels = np.empty((0,2), float)
        
        running_loss = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True).long()
                task_idx_batch = batch['task_idx'].to(device) 
                task_idx = task_idx_batch[0].item() 

                outputs_batch = model(inputs_batch, task_idx) # (batch_size,2) 
                loss_batch = loss_fn(outputs_batch, labels_batch[:,1])
                running_loss.append(loss_batch.item())

                outputs_batch = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch, axis=0)

                labels_batch = labels_batch.cpu().numpy()
                labels = np.append(labels, labels_batch, axis=0)
        
        avg_loss = np.mean(np.array(running_loss))
        
        ## check if its 1st epoch
        if len(self.val_loss_list_dict[tidx]) < 1: ## 1st epoch
            previous_loss = None
        else:
            previous_loss = self.val_loss_list_dict['min'][tidx] ###
        patience = self.task_patiences[tidx]
        epoch_done = current_epoch+1
        
        if previous_loss is None: ## 1st epoch
            self.val_loss_list_dict['min'][tidx] = avg_loss 
        else:
            if avg_loss < previous_loss:
                self.task_patiences[tidx] = 0
                self.val_loss_list_dict['min'][tidx] = avg_loss
            else:
                patience += 1
                self.task_patiences[tidx] = patience
                
                ## if patience reaches threshold, stop and record
                if patience == self.hypers['patience']:
                    self.task_stop_bool[tidx] = True
                    epoch_done = current_epoch+1
                    print_and_log(f'\ttask{tidx} done after {epoch_done} epochs')
                    self.task_stop_dict[epoch_done].append(tidx)

            
        ## hold epoch loss
        self.val_loss_list_dict[tidx].append(avg_loss)

        epoch_auc = self.calculate_auc(logits, labels)
        self.val_auc_list_dict[tidx].append(epoch_auc)
        
            
    def predict_single(self, test_loader, task_idx, epoch=None) -> dict:
        self.check_model()
        model = self.model
        model.eval()
        loss_fn = self.val_loss_fn
        device = self.hypers['device']
        num_tasks = self.hypers['num_tasks']
        
        # create the test result container
        logits = np.empty((0,2), float)
        labels = np.empty((0,2), float)
        
        running_loss = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True).long()
                task_idx_batch = batch['task_idx'].to(device) 
                task_idx = task_idx_batch[0].item() 

                outputs_batch = model(inputs_batch, task_idx) # (batch_size,2) 
                loss_batch = loss_fn(outputs_batch, labels_batch[:,1])
                running_loss.append(loss_batch.item())
                
                outputs_batch = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch, axis=0)
                
                labels_batch = labels_batch.cpu().numpy()
                labels = np.append(labels, labels_batch, axis=0)
        
        
        if not len(running_loss) == 0:
            avg_loss = np.mean(np.array(running_loss))
        else:
            avg_loss = None
            
        self.test_loss_dict[task_idx] = avg_loss
        
        ## calculate auc
        avg_auc = self.calculate_auc(logits, labels)
        self.test_auc_dict[task_idx] = avg_auc
        
        self.save_single_predictions(logits, task_idx, teacher=False) 
    
    
    def predict(self, test_loaders: list, epoch=None):
        self.check_model()
        num_tasks = self.hypers['num_tasks']
        
        test_results = dict()
        for tidx in range(num_tasks):
            test_loader = test_loaders[tidx]
            task_result = self.predict_single(test_loader, tidx, epoch)


    def teacher_predict_single(self, teacher_loader, tidx):
        self.check_model() 
        model = self.model
        model.eval()
#         loss_fn = self.loss_fn
        device = self.hypers['device']
        
        logits = np.empty((0,2), float)
        with torch.no_grad():
            for step, batch in enumerate(teacher_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True)
                task_idx_batch = batch['task_idx'].to(device) 
                task_idx = task_idx_batch[0].item() 

                outputs_batch = model(inputs_batch, task_idx) # (batch_size,2)
                
                outputs_batch = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch, axis=0)
        
        self.save_single_predictions(logits, task_idx) 
            

    def teacher_predict(self, teacher_loaders: list):
        self.check_model()
        num_tasks = self.hypers['num_tasks']
        
        test_results = dict()
        for tidx in range(num_tasks):
            test_loader = teacher_loaders[tidx]
            task_result = self.teacher_predict_single(test_loader, tidx)
        
    
    def train_and_predict(self, train_loaders: list, val_loaders: list, test_loaders: list, teacher_loaders: list=None):
        self.check_model()
        hypers = self.hypers
        device = hypers['device']
        num_epochs = hypers['epochs']
        num_tasks = hypers['num_tasks']
        
        if not teacher_loaders is None:
            teacher = True
        else:
            teacher = False
        
        print_and_log('train')
        for epoch in range(num_epochs):
            print_and_log(f'\tepoch {epoch}')
            
            if np.sum(self.task_stop_bool) == num_tasks:
                print_and_log(f'\tall tasks stop')
                break
                
            self.task_stop_dict[epoch+1] = []
            
            self.train_single_epoch(train_loaders, epoch)
            self.val_single_epoch(val_loaders, epoch)
            
            if len(self.task_stop_dict[epoch+1]) == 0:
                pass
            else:
#                 print_and_log(self.task_stop_dict[epoch+1])
                for task_idx in self.task_stop_dict[epoch+1]:
                    task_test_loader = test_loaders[task_idx]
                    if task_test_loader is not None:
                        self.predict_single(task_test_loader, task_idx)
                    self.task_stop_bool[task_idx] = True
                    
                    if teacher:
                        teacher_loader = teacher_loaders[task_idx]
                        self.teacher_predict_single(teacher_loader, task_idx)     
        
        print_and_log(f'\t{np.sum(self.task_stop_bool == False)} tasks left')
        for task_idx in np.where(self.task_stop_bool == False)[0]:
            task_test_loader = test_loaders[task_idx]
            if task_test_loader is not None:
                self.predict_single(task_test_loader, task_idx)
            self.task_stop_bool[task_idx] = True
            if teacher:
                teacher_loader = teacher_loaders[task_idx]
                self.teacher_predict_single(teacher_loader, task_idx)
            
        assert np.sum(self.task_stop_bool) == num_tasks, "there are some missing task"
        
        print_and_log('save')
        for tidx in range(num_tasks):
            self.save_single_containers(tidx)
            task_train_dataset = train_loaders[0].dataset.taskidx2dataset[tidx].dataset
            if teacher:
                task_teacher_dataset = teacher_loaders[tidx].dataset.dataset
            

    def calculate_auc(self, logits, labels):
        if labels.shape[1] == 2:
            labels = labels[:,1]
        
        try:
            scores = special.softmax(logits, axis=1)
            auc_score = metrics.roc_auc_score(labels, scores[:,1])
            return auc_score
        ### if there is only one class in label
        except ValueError:
            return -1
               

    def save_single_predictions(self, predictions, tidx, teacher=False):
        dname = f'task{tidx:0>2}'
        task_dir = os.path.join(self.dirs['cluster_dir'], dname)
        if_mkdir(task_dir)
        self.dirs['task_dir'][tidx] = task_dir
        
        ## test logit
        if not teacher: 
             lname = 'test_logit.npy'
        ## teacher logit
        else: 
            lname = 'teacher_logit.npy'
        
        path = os.path.join(task_dir, lname)
        np.save(path, predictions)
    
    
    def save_single_containers(self, tidx):
        dname = f'task{tidx:0>2}'
        task_dir = os.path.join(self.dirs['cluster_dir'], dname)
        if_mkdir(task_dir)
        self.dirs['task_dir'][tidx] = task_dir
        task_dir = self.dirs['task_dir'][tidx]

        train_path = os.path.join(task_dir, f'train_loss.npy')
        np.save(train_path, np.asarray(self.train_loss_list_dict[tidx]))
        train_path = os.path.join(task_dir, f'train_auc.npy')
        np.save(train_path, np.asarray(self.train_auc_list_dict[tidx]))
        
        val_path = os.path.join(task_dir, f'val_loss.npy')
        np.save(val_path, np.asarray(self.val_loss_list_dict[tidx]))
        val_path = os.path.join(task_dir, f'val_auc.npy')
        np.save(val_path, np.asarray(self.val_auc_list_dict[tidx]))
        
        test_path = os.path.join(task_dir, f'test_results.pickle')
        test_result = {'loss': self.test_loss_dict[tidx], 'auc': self.test_auc_dict[tidx]}
        with open(test_path, 'wb') as wf:
            pickle.dump(test_result, wf)
 

        
