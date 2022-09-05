import CMAPSSDataset

window_size = 32
datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=32, sequence_length=window_size)
max_life_time = datasets.max_life_time
train_data = datasets.get_train_data()
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)/max_life_time
test_data = datasets.get_test_data()
test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
test_label_slice = test_label_slice/max_life_time
sequence_x,sequence_y = datasets.get_nid_sequence(train_data,19)
sequence_y = sequence_y/max_life_time