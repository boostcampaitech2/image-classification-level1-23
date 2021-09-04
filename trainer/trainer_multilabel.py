import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold , StratifiedKFold
from tqdm import tqdm


def func_acc(outputs, labels):
    cnt_answer = 0
    for i in range(len(labels)):
        label = labels[i]
        _, output = outputs[i].max(dim=0)
        if label == output:
            cnt_answer += 1
        
    return cnt_answer / len(labels)
    
def func_class_acc(outputs, labels, pre_acc_list):
    # return: [[True, False, ], [], [], ...]
    answer = [[] for i in range(18)]    
    
    is_answer = [[] for i in range(18)]
    for i in range(len(labels)):
        label = labels[i]
        _, output = outputs[i].max(dim=0)
        if label == output:
            is_answer[label].append(True)
        else:
            is_answer[label].append(False)

    for i in range(18):
        answer[i] = pre_acc_list[i] + is_answer[i]           
    return answer

def func_class_acc_mask(outputs, labels, pre_acc_list):
    # return: [[True, False, ], [], [], ...]
    answer = [[] for i in range(3)]    
    
    is_answer = [[] for i in range(3)]
    for i in range(len(labels)):
        label = labels[i]
        _, output = outputs[i].max(dim=0)
        if label == output:
            is_answer[label].append(True)
        else:
            is_answer[label].append(False)
            
    for i in range(3):
        answer[i] = pre_acc_list[i] + is_answer[i]
                           
    return answer

def func_class_acc_gender(outputs, labels, pre_acc_list):
    # return: [[True, False, ], [], [], ...]
    answer = [[] for i in range(2)]    
    
    is_answer = [[] for i in range(2)]
    for i in range(len(labels)):
        label = labels[i]
        _, output = outputs[i].max(dim=0)
        if label == output:
            is_answer[label].append(True)
        else:
            is_answer[label].append(False)
            
    for i in range(2):
        answer[i] = pre_acc_list[i] + is_answer[i]
                           
    return answer

def func_class_acc_age(outputs, labels, pre_acc_list):
    # return: [[True, False, ], [], [], ...]
    answer = [[] for i in range(3)]    
    
    is_answer = [[] for i in range(3)]
    for i in range(len(labels)):
        label = labels[i]
        _, output = outputs[i].max(dim=0)
        if label == output:
            is_answer[label].append(True)
        else:
            is_answer[label].append(False)
            
    for i in range(3):
        answer[i] = pre_acc_list[i] + is_answer[i]                   
    return answer

def cal_class_acc(epoch_class_acc):
    output = [0] * 18
    for i in range(18):
        total_cnt = len(epoch_class_acc[i])
        answer_cnt = 0
        for answer in epoch_class_acc[i]:
            if answer:
                answer_cnt += 1
        output[i] = answer_cnt / total_cnt
    return output

def cal_class_acc_mask(epoch_class_acc):
    output = [0] * 3
    for i in range(3):
        total_cnt = len(epoch_class_acc[i])
        answer_cnt = 0
        for answer in epoch_class_acc[i]:
            if answer:
                answer_cnt += 1
        output[i] = answer_cnt / total_cnt
    return output

def cal_class_acc_gender(epoch_class_acc):
    output = [0] * 2
    for i in range(2):
        total_cnt = len(epoch_class_acc[i])
        answer_cnt = 0
        for answer in epoch_class_acc[i]:
            if answer:
                answer_cnt += 1
        output[i] = answer_cnt / total_cnt
    return output


def cal_class_acc_age(epoch_class_acc):
    output = [0] * 3
    for i in range(3):
        total_cnt = len(epoch_class_acc[i])
        answer_cnt = 0
        for answer in epoch_class_acc[i]:
            if answer:
                answer_cnt += 1
        output[i] = answer_cnt / total_cnt
    return output

#outputs_label = func_labels(outputs_mask, outputs_gender, outputs_age)
def func_labels(outputs_mask, outputs_gender, outputs_age, device):
    # outputs_label = [class1, class2, class3, ...]
    outputs_label = torch.Tensor([])
    len_outputs = len(outputs_mask)
    for i in range(len_outputs):
        mask_class = outputs_mask[i] # [0.6, 0.2, 0.1, 0.1]
        _, mask_class = mask_class.max(dim=0)
        
        gender_class = outputs_gender[i] # [0.6, 0.2, 0.1, 0.1]
        _, gender_class = gender_class.max(dim=0)
        
        age_class = outputs_age[i] # [0.6, 0.2, 0.1, 0.1]
        _, age_class = age_class.max(dim=0)
        
        label = mask_class * 6 + gender_class * 3 + age_class
        
        #label: int -> [[1, 0, 0, 0]]
        one_hot = torch.zeros((1,18))
        one_hot[0][label] = 1
        label = one_hot
        outputs_label = torch.cat([outputs_label, label])
    return outputs_label.to(device)


class CostumTrain(object):
    def __init__(self, model, config, criterion, optimizer, device, data_set, transform, scheduler):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.data_set = data_set
        self.transform = transform
        self.scheduler = scheduler

    def train(self,):
     
        min_loss = float('inf')

        for epoch in range(self.config["trainer"]["epochs"]):
            epoch_loss = 0
            epoch_acc = 0
            epoch_val_loss = 0
            epoch_val_acc = 0

            epoch_class_acc = [[] for i in range(18)]
            epoch_class_val_acc = [[] for i in range(18)]
            
            epoch_mask_acc = [[] for i in range(18)]
            epoch_mask_val_acc = [[] for i in range(18)]
            
            epoch_gender_acc = [[] for i in range(18)]
            epoch_gender_val_acc = [[] for i in range(18)]
            
            epoch_age_acc = [[] for i in range(18)]
            epoch_age_val_acc = [[] for i in range(18)]
            
            
            stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            k_idx = 1
            for train_index, validate_index in stratified_kfold.split(np.zeros(len(self.data_set.age)), self.data_set.age):  
                print(f'## Stratified_K-Fold :: {k_idx}')
                k_idx += 1
                train_dataset = torch.utils.data.dataset.Subset(self.data_set, train_index)
                valid_dataset = torch.utils.data.dataset.Subset(self.data_set, validate_index)
                valid_dataset = copy.deepcopy(valid_dataset)
                valid_dataset.dataset.transform = self.transform.transformations["val"]
                train_loader = DataLoader(train_dataset,
                            batch_size=self.config["data_loader"]["args"]["batch_size"],
                            shuffle=self.config["data_loader"]["args"]["shuffle"],
                            num_workers=self.config["data_loader"]["args"]["num_workers"],
                            drop_last=True 
                        )
                val_loader = DataLoader(valid_dataset,
                            batch_size=self.config["data_loader"]["args"]["batch_size"],
                            shuffle=self.config["data_loader"]["args"]["shuffle"],
                            num_workers=self.config["data_loader"]["args"]["num_workers"],
                            drop_last=True 
                        )
                
                for i, data in tqdm(enumerate(train_loader), desc=f"epoch-{epoch}", total=len(train_loader)):
                    inputs, (labels, masks, genders, ages) = data

                    self.optimizer.zero_grad()
                    outputs_mask, outputs_gender, outputs_age = self.model(inputs)
                    outputs_label = func_labels(outputs_mask, outputs_gender, outputs_age, self.device)

                    loss_masks = self.criterion(outputs_mask, masks)
                    loss_genders = self.criterion(outputs_gender, genders)
                    loss_ages = self.criterion(outputs_age, ages)

                    loss = loss_masks + loss_genders + loss_ages

                    epoch_loss += loss

                    acc = func_acc(outputs_label, labels)
                    epoch_acc += acc     

                    epoch_class_acc = func_class_acc(outputs_label, labels, epoch_class_acc)

                    epoch_mask_acc = func_class_acc_mask(outputs_mask, masks, epoch_mask_acc)
                    epoch_gender_acc = func_class_acc_gender(outputs_gender, genders, epoch_gender_acc)
                    epoch_age_acc = func_class_acc_age(outputs_age, ages, epoch_age_acc)

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        val_inputs, (val_labels, val_masks, val_genders, val_ages) = data
                        val_outputs_mask, val_outputs_gender, val_outputs_age = self.model(val_inputs)
                        val_outputs_label = func_labels(val_outputs_mask, val_outputs_gender, val_outputs_age, self.device)

                        val_loss_mask = self.criterion(val_outputs_mask, val_masks)
                        val_loss_gender = self.criterion(val_outputs_gender, val_genders)
                        val_loss_age = self.criterion(val_outputs_age, val_ages)
                        val_loss = val_loss_mask + val_loss_gender + val_loss_age

                        epoch_val_loss += val_loss

                        val_acc = func_acc(val_outputs_label, val_labels)
                        epoch_val_acc += val_acc

                        epoch_class_val_acc = func_class_acc(val_outputs_label, val_labels, epoch_class_val_acc)
                        epoch_mask_val_acc = func_class_acc_mask(val_outputs_mask, val_masks, epoch_mask_val_acc)
                        epoch_gender_val_acc = func_class_acc_gender(val_outputs_gender, val_genders, epoch_gender_val_acc)
                        epoch_age_val_acc = func_class_acc_age(val_outputs_age, val_ages, epoch_age_val_acc)



            epoch_loss /= len(train_loader) * 5
            epoch_acc /= len(train_loader) * 5
            epoch_class_acc = cal_class_acc(epoch_class_acc)
            # print(f"eporch_mask_acc {epoch_mask_acc}")
            epoch_mask_acc = cal_class_acc_mask(epoch_mask_acc)
            epoch_gender_acc = cal_class_acc_gender(epoch_gender_acc)
            epoch_age_acc = cal_class_acc_age(epoch_age_acc)

            epoch_val_loss /= len(val_loader) * 5
            epoch_val_acc /= len(val_loader) * 5
            epoch_class_val_acc = cal_class_acc(epoch_class_val_acc)
            epoch_mask_val_acc = cal_class_acc_mask(epoch_mask_val_acc)
            epoch_gender_val_acc = cal_class_acc_gender(epoch_gender_val_acc)
            epoch_age_val_acc = cal_class_acc_age(epoch_age_val_acc)
            
            if min_loss > epoch_loss:
                save_path = f'./save_model/epoch_{epoch+1}_loss_{epoch_loss}.pth'
                torch.save(self.model.state_dict(), save_path)
                min_loss = epoch_loss
            
            print(f'epoch: {epoch}, epoch_acc: {epoch_acc}, epoch_loss: {epoch_loss}')
            
            print(f'epoch: {epoch}, epoch_val_acc: {epoch_val_acc}, epoch_val_loss: {epoch_val_loss}')    
            print(f'epoch_class_acc:')
            for class_id in range(18):
                print(f'class{class_id}: {epoch_class_acc[class_id]:.3f}(train_label) / {epoch_class_val_acc[class_id]:.3f}(val_label)')
            
            print(f'\nepoch_mask_acc:')
            for class_id in range(3):
                print(f'class{class_id}: {epoch_mask_acc[class_id]:.3f}(train_mask) / {epoch_mask_val_acc[class_id]:.3f}(val_mask)')
            
            print(f'\nepoch_gender_acc:')
            for class_id in range(2):
                print(f'class{class_id}: {epoch_gender_acc[class_id]:.3f}(train_gender) / {epoch_gender_val_acc[class_id]:.3f}(val_gender)')                
            
            print(f'\nepoch_age_acc:')
            for class_id in range(3):
                print(f'class{class_id}: {epoch_age_acc[class_id]:.3f}(train_age) / {epoch_age_val_acc[class_id]:.3f}(val_age)')   
