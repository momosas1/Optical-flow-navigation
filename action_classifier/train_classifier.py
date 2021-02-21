import torch
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader
from .classifier_net import Classifier
from .flowaction_dataset import trainset
import torch.nn as nn



def main():
    '''
    train flow classifier use two flow pictures as input
    if want to train the network with single flow picture need to fix the dataset getitem() just use flow_c list
    '''
    net = Classifier()
    device = torch.device("cuda:0")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0001 , betas=(0.9,0.99))

    running_loss  = 0.0
    steps = 0
    success = 0
    train_data = trainset()
    trainloader = DataLoader(train_data, batch_size=4, shuffle=True)


    ckpt = torch.load("flow_data/flow_ckpt/ckpt_flow.pth")
    net.load_state_dict(ckpt['state_dict'])

    for j in range(100):

        for i, data in enumerate(trainloader, 0):
            flow, action = data
            flow = flow.to(device)

            action = action.numpy()
            action_list = []

            #proccess the data shape
            for i in range(len(action)):
                action_list.append(action[i][0])
            action_list = torch.from_numpy(np.array(action_list))
            action_list = action_list.to(device)


            optimizer.zero_grad()
            outputs = net(flow)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, action_list)
            loss.backward()
            optimizer.step()

            for i in range(len(predicted)):
                if predicted[i] == action_list[i]:
                    success+= 1

            running_loss += loss.item()
            steps += len(predicted)

        if j % 5 == 0:
            print('[%d, %5d] loss: %.3f     success:%.3f' %
                  (j + 1, steps, running_loss / steps, success / steps))
            running_loss = 0.0
            steps = 0
            success = 0
            checkpoint = {"state_dict": net.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    "ck_act",
                    "ckpt_flow_single.pth",
                ),
            )


if __name__ == "__main__":
    main()