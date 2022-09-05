import CMAPSSDataset
import numpy as np

datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=32, sequence_length=32)
train_data = datasets.get_train_data()
sequence_x,sequence_y = datasets.get_nid_sequence(train_data,1)
print(sequence_x.shape)
print(sequence_y)
