import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics
from data_stream import DataStream
from sklearn import preprocessing
from tcn import TemporalConvNet

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
import matplotlib.pyplot as plt

def pre_process(data_ori):
    data_ori = preprocessing.scale(np.reshape(data_ori, [-1, 2]))
    processed_data = np.reshape(data_ori, [-1, 32768, 2])
    return processed_data

condition=1
time_step=32
#s2o 堆叠时间窗+选取一个发动机
#此处未实现分段函数RUL
bearing1 = pre_process(np.load('data/Bearing{}_1.npy'.format(condition)))
train_instances = [bearing1] #4维 （1，分钟数，32768，2）
test_instances = [bearing1]  #暂且用同一个集合
#实际上train 1,2,3,4 test5
max_bearing_lifetime = max([instance.shape[0] for instance in train_instances+test_instances])#所有轴承所有状况生命最长的
# 除以最长寿命
data_stream_train = DataStream(train_instances, time_step, True, max_bearing_lifetime)
data_stream_valid = DataStream(test_instances, time_step, False, max_bearing_lifetime)
max_life_time = data_stream_valid.max_bearing_lifetime
train_x,train_y = data_stream_train.instances
test_x,test_y = data_stream_valid.instances

batch_num,sequence_length,minute_sampe,feature_dim=np.array(train_x).shape

X_train = torch.tensor(train_x).float()
y_train = torch.tensor(train_y).float().unsqueeze(-1)
X_test = torch.tensor(test_x).float()
y_test = torch.tensor(test_y).float().unsqueeze(-1)

ds = torch.utils.data.TensorDataset(X_train,y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.tcn =TemporalConvNet(minute_sampe*feature_dim, [32,16,8,4], kernel_size=2, dropout=0)
        self.lstm_1 = BayesianLSTM(4, 1, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)

    def forward(self, x):
        x = x.reshape(-1,sequence_length,minute_sampe*feature_dim)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)

        x_, _ = self.lstm_1(x)
        out = x_[:, -1, :]
        return out
net = NN()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) #0.001感觉还是有点大

#train
iteration = 0
for epoch in range(4):#test
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = net.sample_elbo(inputs=datapoints,
                               labels=labels,
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=1 / X_train.shape[0])
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 10 == 0:
            preds_test = net(X_test)
            loss_test = criterion(preds_test, y_test)
            print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))

#评估
def pred_future(test,sample_nbr):
    preds_test = []
    for i in range(len(test)):
        input = test[i].unsqueeze(0)
        pred = [net(input).cpu().item() for i in range(sample_nbr)]
        preds_test.append(pred)
    preds_test = np.array(preds_test)
    return preds_test

def get_confidence_intervals(preds_test, ci_multiplier):
    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(1)
    pred_std = preds_test.std(1).detach().cpu().numpy()

    pred_std = torch.tensor((pred_std))

    upper_bound = (pred_mean + (pred_std * ci_multiplier))
    lower_bound = (pred_mean - (pred_std * ci_multiplier))
    return pred_mean,upper_bound,lower_bound

preds_test=pred_future(X_test,5)*max_life_time
pred_mean,upper_bound,lower_bound=get_confidence_intervals(preds_test,5)
y_true = np.array(test_y)*max_life_time
y = np.arange(len(y_true))
print(np.array(pred_mean).shape)
print(np.array(y_true).shape)

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

params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}

plt.rcParams.update(params)

plt.title("XJBearing_s2o", color="white")

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

plt.legend()
plt.show()