#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:44:20 2020

@author: yunda_si
"""

from ResNetB import resnet18
import torch
import torch.optim as optim
import time
import random
from ppi_loss import ppi_loss
import pickle
import os
import numpy as np
from top_statistics import top_statistics_ppi
from ppi_dataloader import PPI_Dataset
from torch.utils.data import DataLoader


def concat(A_f1d, B_f1d, p2d):
    
    def rep_new_axis(mat, rep_num, axis):
        return torch.repeat_interleave(torch.unsqueeze(mat,axis=axis),rep_num,axis=axis)
    
    len_channel,lenA = A_f1d.shape
    len_channel,lenB = B_f1d.shape        
    
    row_repeat = rep_new_axis(A_f1d, lenB, 2)
    col_repeat = rep_new_axis(B_f1d, lenA, 1)        

    return  torch.unsqueeze(torch.cat((row_repeat, col_repeat),axis=0),0)


random.seed(42)
###################              load dataset               ###################
homo_path = '/mnt/data/yunda_si/self/data/Homodataset/train_paired'
hetero_path = '/mnt/data/yunda_si/self/data/Heterodataset/train_paired/'
train_all_path = '/mnt/data/yunda_si/self/data/train_ppi/'

hetero_lists = sorted(os.listdir(hetero_path))
homo_lists = sorted(os.listdir(homo_path))
all_lists = sorted(os.listdir(train_all_path))

for i in range(10):
    random.shuffle(hetero_lists)
    random.shuffle(homo_lists)    
    random.shuffle(all_lists)
    
train_list = hetero_lists[500:]#+all_lists[6300:]
valid_list = hetero_lists[:500]

trainset = PPI_Dataset(train_all_path, train_list)
train_loader = DataLoader(trainset, shuffle=True, num_workers=6, prefetch_factor=3, 
                          batch_size=None, persistent_workers=True)

validset = PPI_Dataset(train_all_path, valid_list)
valid_loader = DataLoader(validset, shuffle=True, num_workers=6, prefetch_factor=3,
                          batch_size=None, persistent_workers=True)

max_aa = 400

###################               import net                ###################
device = torch.device("cuda:1")
print(device)
model = resnet18().to(device)
# model.load_state_dict(torch.load('/mnt/data/yunda_si/self/PythonProjects/PPI_contact/train_v2/model/basemodel_all_feature_v8_1.pth'))
criterion_ppi = ppi_loss(alpha=None, reduction='sum')
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                        weight_decay=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                        eps=1e-6, patience=1, factor=0.1, verbose=True)

epoch_num = 25


###################             top statistics              ###################
topk_ppi = ['L/5','L/10','L/20',50,20,10,5,1]
dict_statics = {'min_loss':np.inf,'valid_loss':[]}

for key in topk_ppi:
    dict_statics[key] = {'highest':0,'save':'','train_acc':[],'valid_acc':[]}


###################               save model                ###################
if os.path.exists('final_model'):
    pass
else:
    os.mkdir('final_model')

savepth = './final_model/PSSM_'

for key in topk_ppi:
    dict_statics[key]['save'] = '{0}_{1}.pth'.format(savepth, str(key).replace('/','_'))
loss_save = f'{savepth}_minloss.pth'


###################                training                 ###################
for epoch in range(epoch_num):
    since = time.time()

    print('learning rate: %8.6f' %optimizer.param_groups[0]['lr'])

    for phase in ['train', 'valid']:
        print('\n')
        if phase == 'train':
            model.train()
            dataloader = train_loader
            acc_all = np.zeros((0,len(topk_ppi)))
        else:
            model.eval()
            dataloader = valid_loader
            acc_all = np.zeros((0,len(topk_ppi)))

        running_loss = 0.0
        optimizer.zero_grad()

        for d, (_, A_f1d, B_f1d, p2d, mask_map, contact_map) in enumerate(dataloader):


            A_f1d = A_f1d.to(device).squeeze().float()
            B_f1d = B_f1d.to(device).squeeze().float()
            # p2d = torch.cat([i.to(device) for i in p2d],axis=1).squeeze().float()
            mask_map = mask_map.squeeze().to(device).float()
            contact_map = contact_map.squeeze().to(device).float()
            
            la,lb = contact_map.shape
            starta = 0 if la<=max_aa else np.random.randint(0,la-max_aa+1)
            startb = 0 if lb<=max_aa else np.random.randint(0,lb-max_aa+1)   
            
            A_f1d = A_f1d[:, starta:(starta+max_aa)]
            B_f1d = B_f1d[:, startb:(startb+max_aa)]
            # p2d = p2d[:, starta:(starta+max_aa), startb:(startb+max_aa)]
            mask_map = mask_map[starta:(starta+max_aa), startb:(startb+max_aa)]
            contact_map = contact_map[starta:(starta+max_aa), startb:(startb+max_aa)]

        
            Input = concat(A_f1d, B_f1d, p2d)
            # Input = torch.unsqueeze(p2d,0)#concat(A_f1d, B_f1d, p2d)
        
            
            with torch.set_grad_enabled(phase == 'train'):
                preds = model(Input)
                loss = criterion_ppi(preds, contact_map, mask_map)

                if phase == 'train':
                    loss = loss
                    loss.backward()
                    
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item()

            ##################          statistics           ##################
            accuracy = top_statistics_ppi(preds,contact_map,topk_ppi)
            acc_all = np.vstack([acc_all,accuracy])
            
            
            if (d+1)%100==0:
                mean_acc = np.mean(acc_all,0)*100
                print(f'[{epoch:3d}, {d+1:4d}]  loss:{running_loss:11.2f} {"  ".join([f"{i.item():7.3f}" for i in mean_acc])}')
            if (d+1)==len(dataloader):
                mean_acc = np.mean(acc_all,0)*100
                print(f'[{epoch:3d}, {d+1:4d}]  loss:{running_loss:11.2f} {"  ".join([f"{i.item():7.3f}" for i in mean_acc])}')


        if phase == 'valid':
            scheduler.step(running_loss)
            dict_statics['valid_loss'].append(running_loss)
            for index,key in enumerate(topk_ppi):
                dict_statics[key]['valid_acc'].append(mean_acc[index])
        else:
            for index,key in enumerate(topk_ppi):
                dict_statics[key]['train_acc'].append(mean_acc[index])            
            
    ##################                 save                  ##################
    for key in topk_ppi:
        acc = dict_statics[key]['valid_acc'][-1]
        highest = dict_statics[key]['highest']
        if acc>highest:
            print(f'save_{str(key):5s}:{acc:6.3f}  highest: {highest:6.3f}  delta:{acc-highest:6.3f}')
            dict_statics[key]['highest'] = acc

            if os.path.exists(dict_statics[key]['save']):
                os.remove(dict_statics[key]['save'])
            torch.save(model.state_dict(), dict_statics[key]['save'])

    if running_loss<dict_statics['min_loss']:
        print('save_minloss:%11.2f    %11.2f'%(running_loss,dict_statics['min_loss']))
        dict_statics['min_loss'] = running_loss
        torch.save(model.state_dict(), loss_save)

    file = open('%s_dict_statics.pkl'%(savepth), 'wb')
    pickle.dump(dict_statics, file)
    file.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

print('Finished Training')
highest_list = ['%s: %6.3f'%(i, dict_statics[i]['highest']) for i in topk_ppi]
print('highest:%s'%'  '.join(highest_list))












