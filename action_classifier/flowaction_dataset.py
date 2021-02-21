import os
import cv2
import torch
from torch.utils.data import Dataset

class trainset(Dataset):
    '''
    flow action dataset
    '''
    def __init__(self, loader=None):
        '''
        filelist_c is current flow picture folder
        filelist_p is last flow picture folder
        the picture name is action type
        '''
        filelist_c = os.listdir("flow_data/flow_c")
        filelist_c.sort()
        filelist_p = os.listdir("flow_data/flow_p")
        filelist_p.sort()

        file_train_c=[]
        file_train_p=[]
        number_train=[]

        for i in range(len(filelist_c)):

            file_train_c.append(os.path.join("flow_data/flow_c",filelist_c[i]))
            file_train_p.append(os.path.join("flow_data/flow_p",filelist_p[i]))
            number_train.append(int(filelist_c[i][6]))

        self.images_c = file_train_c
        self.images_p = file_train_p
        self.target = number_train
        self.loader = loader


    def __getitem__(self, index):
        fn = cv2.imread(self.images_c[index])
        img_c = fn/255.0
        img_c = torch.Tensor(img_c)

        fn = cv2.imread(self.images_p[index])
        img_p = fn / 255.0
        img_p = torch.Tensor(img_p)

        target = self.target[index]
        target = torch.Tensor([target]).long()

        img = torch.cat([img_c.permute(2, 0, 1),img_p.permute(2, 0, 1)],dim=0)
        #img = img_c.permute(2, 0, 1)
        return img,target

    def __len__(self):
        return len(self.images_c)