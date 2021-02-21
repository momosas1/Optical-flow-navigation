import numpy as np
import cv2
import torch


def flow_to_rgb(flow):
    flow1=flow.numpy().transpose(2,0,1)
    tflow=np.minimum(20,(np.maximum(-20,flow1)))
    iflow = ( (tflow+20)/40.0 ).astype(np.uint8)
    return iflow.transpose(1,2,0)