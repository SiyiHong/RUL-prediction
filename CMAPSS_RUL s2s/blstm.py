import CMAPSSDataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from tcn import TemporalConvNet

import torch
import torch.nn as nn
import torch.optim as optim

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

window_size = 32
datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=32, sequence_length=window_size)
max_life_time = datasets.max_life_time
train_data = datasets.get_train_data()
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)/max_life_time
test_data = datasets.get_test_data()
test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
test_label_slice = test_label_slice/max_life_time
sequence_x,sequence_y = datasets.get_nid_sequence(train_data,11)
sequence_y = sequence_y/max_life_time

timesteps = train_feature_slice.shape[1]
input_dim = train_feature_slice.shape[2]

X_train = torch.tensor(train_feature_slice).float()
y_train = torch.tensor(train_label_slice).float()
X_test = torch.tensor(test_feature_slice).float()
y_test = torch.tensor(test_label_slice).float()
test_x_seq = torch.tensor(sequence_x).float()
test_y_seq = torch.tensor(sequence_y).float()

ds = torch.utils.data.TensorDataset(X_train,y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)  #可能也要改成32

@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        #self.lstm_1 = BayesianLSTM(25, 1, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.lstm_1 = nn.LSTM(25, 1)

    def forward(self, x):
        x_, _ = self.lstm_1(x)
        out = np.squeeze(x_, axis=-1)
        return out

net = NN()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001) #学习率有点大

#train
iteration = 0
# for epoch in range(3):#test
#     for i, (datapoints, labels) in enumerate(dataloader_train):
#         optimizer.zero_grad()
#
#         loss = net.sample_elbo(inputs=datapoints,
#                                labels=labels,
#                                criterion=criterion,
#                                sample_nbr=3,
#                                complexity_cost_weight=1 / X_train.shape[0])
#         loss.backward()
#         optimizer.step()
#
#         iteration += 1
#         if iteration % 250 == 0:
#             preds_test = net(X_test)
#             loss_test = criterion(preds_test, y_test)
#             print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))
#
# torch.save(net.state_dict(), 'best44.pt')
net.load_state_dict(torch.load('best5.pt'))

def pred_future(test,sample_nbr):
    preds_test = []
    for i in range(len(test)):
        input = test[i].unsqueeze(0)
        pred = np.stack([net(input).cpu().detach().numpy().squeeze() for i in range(sample_nbr)],axis=1)
        preds_test.extend(pred)
    preds_test=np.array(preds_test)
    return preds_test

#评估
def get_confidence_intervals(preds_test, ci_multiplier):
    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(1)
    pred_std = preds_test.std(1).detach().cpu().numpy()

    pred_std = torch.tensor((pred_std))

    upper_bound = (pred_mean + (pred_std * ci_multiplier))
    lower_bound = (pred_mean - (pred_std * ci_multiplier))
    return pred_mean,upper_bound,lower_bound,pred_std

# preds_test=pred_future(X_test,5)*max_life_time
# pred_mean,upper_bound,lower_bound = get_confidence_intervals(preds_test,2)
# y_true = y_test.reshape(-1)*max_life_time
# y = np.arange(len(y_true))

preds_test = pred_future(test_x_seq,5)*max_life_time
pred_mean,upper_bound,lower_bound,std = get_confidence_intervals(preds_test,2)
y_true = test_y_seq.reshape(-1)*max_life_time
y = np.arange(len(y_true))

explained_variance=sklearn.metrics.explained_variance_score(y_true,pred_mean)
mse = sklearn.metrics.mean_squared_error(y_true,pred_mean)
mae = sklearn.metrics.mean_absolute_error(y_true,pred_mean)
medae=sklearn.metrics.median_absolute_error(y_true,pred_mean)
R2 = sklearn.metrics.r2_score(y_true,pred_mean)
print('explained variance:',explained_variance)  #保留n 位小数：round( ,n)
print('MSE:',mse)
print('MAE:',mae)
print('MedAE:',medae)
print('R2:',R2)

# plt.subplot(1,2,1)
plt.plot(y,
         y_true,
         color='black',
         label="Real")

plt.plot(y,
         pred_mean,
         label="Prediction",
         color="red")

# plt.fill_between(x=y,
#                  y1=upper_bound,
#                  y2=lower_bound,
#                  facecolor='green',
#                  label="Confidence interval",
#                  alpha=0.5)
# plt.subplot(1,2,2)
# plt.plot(
#     y,std,color='blue',label='std'
# )
plt.legend()
plt.savefig('./lstm_sequence')

