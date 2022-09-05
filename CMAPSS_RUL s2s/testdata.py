import CMAPSSDataset

import numpy as np
import torch
window_size = 32
datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=32, sequence_length=window_size)
train_data = datasets.get_train_data()

features,labels = datasets.get_nid_sequence(train_data,19)





