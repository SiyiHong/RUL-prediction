import CMAPSSDataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torch.nn as nn
import torch.optim as optim
from blitz.utils import variational_estimator
from blitz.modules import BayesianLSTM

window_size = 32
datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=32, sequence_length=window_size)
train_data = datasets.get_train_data()
max_life_time = datasets.max_life_time
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)
test_data = datasets.get_test_data()
test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
sequence_x,sequence_y = datasets.get_nid_sequence(train_data,88) #num_id

timesteps = train_feature_slice.shape[1]
input_dim = train_feature_slice.shape[2]

X_train = torch.tensor(train_feature_slice).float()
y_train = torch.tensor(train_label_slice).float()
X_test = torch.tensor(test_feature_slice).float()
y_test = torch.tensor(test_label_slice).float()
X_seq_test = torch.tensor(sequence_x).float()
y_seq_test = torch.tensor(sequence_y).float()


ds = torch.utils.data.TensorDataset(X_train,y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        #self.lstm_1 = nn.LSTM(25, 50)
        self.lstm_1 = BayesianLSTM(25, 50, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x_, _ = self.lstm_1(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_
net = NN()

criterion = nn.MSELoss() #MAPE
optimizer = optim.Adam(net.parameters(), lr=0.001)
# train
iteration = 0

# for epoch in range(5):#test
#     for i, (datapoints, labels) in enumerate(dataloader_train):
#         optimizer.zero_grad()
#         out = net(datapoints)
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
#             preds_test = net(X_test)[:, 0].unsqueeze(1)
#             loss_test = criterion(preds_test, y_test)
#             print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))
# torch.save(net.state_dict(), 'best4.pt')

net.load_state_dict(torch.load('best3.pt'))
def pred_future(test,sample_nbr):
    preds_test = []
    for i in range(len(test)):
        input = test[i].unsqueeze(0)
        pred = [net(input).cpu().item() for i in range(sample_nbr)]
        preds_test.append(pred)
    return preds_test

def get_confidence_intervals(preds_test, ci_multiplier):
    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(1)
    pred_std = preds_test.std(1).detach().cpu().numpy()

    pred_std = torch.tensor((pred_std))

    upper_bound = (pred_mean + (pred_std * ci_multiplier))
    lower_bound = (pred_mean - (pred_std * ci_multiplier))
    return pred_mean,upper_bound,lower_bound,pred_std

preds_test=pred_future(X_seq_test,5)
pred_mean,upper_bound,lower_bound,std = get_confidence_intervals(preds_test,2)
result = np.array(pred_mean)
# np.savetxt('result.txt', result, fmt='%d', delimiter=',')
# print(result)
y_true = y_seq_test.reshape(-1)
y=np.arange(len(y_true))
# under_upper = upper_bound > y_true
# over_lower = lower_bound < y_true
# total = np.array(under_upper == over_lower)
# print('between CI:',np.mean(total))
# preds_test = pred_future(X_seq_test,5)
# pred_mean,upper_bound,lower_bound,std=get_confidence_intervals(preds_test,2)
# y_true = y_seq_test.reshape(-1)
# y=np.arange(len(y_true))

#??????
explained_variance=sklearn.metrics.explained_variance_score(y_true,pred_mean)
mse = sklearn.metrics.mean_squared_error(y_true,pred_mean)
mae = sklearn.metrics.mean_absolute_error(y_true,pred_mean)
medae=sklearn.metrics.median_absolute_error(y_true,pred_mean)
R2 = sklearn.metrics.r2_score(y_true,pred_mean)
print('explained variance:',explained_variance)  #??????n ????????????round( ,n)
print('MSE:',mse)
print('MAE:',mae)
print('MedAE:',medae)
print('R2:',R2)

plt.subplot(1,2,1)
plt.plot(y,
         y_true,
         color='black',
         label="Real")

plt.plot(y,
         pred_mean,
         label="Prediction",
         color="red")

plt.fill_between(x=y,
                 y1=upper_bound,
                 y2=lower_bound,
                 facecolor='green',
                 label="Confidence interval",
                 alpha=0.5)
# plt.legend()
# plt.show()
plt.subplot(1,2,2)
plt.plot(
    y,std,color='blue',label='std'
)
plt.legend()
plt.savefig('./btcn_sequence1')