import numpy as np

class DataStream(object):
    def __init__(self,
                 instances,
                 time_step,
                 is_training,
                 max_bearing_lifetime):
        self.max_bearing_lifetime = max_bearing_lifetime
        self.time_step = time_step
        self.bearing_lifetimes = [instance.shape[0]
                                  for instance in instances]  #所有轴承的生命周期

        self.instances = self.split_instances(
            instances, is_training, time_step) #四维 （划分的组数，time_step，每分钟采样数32768,特征维度）
        self.nb_instances = len(self.instances)

    # split the long instance to given number samples with fixed time step
    def split_instances(self, instances, is_training, time_step):
        X = []
        Y = []
        for i in range(len(instances)): #对每一个轴承
            input_all = instances[i]
            #改分段模型的话在此处改
            y_true = np.arange(self.bearing_lifetimes[i])[::-1] /\
                self.max_bearing_lifetime  #求剩余寿命并归一
            if is_training:
                for j in range(self.bearing_lifetimes[i]-time_step):
                    X.append(input_all[j:j+time_step, :, :])
                    Y.append(y_true[j:j+time_step])
                    #  add the tail data
                    if j+40 > self.bearing_lifetimes[i]-time_step:
                        X += 2 * [input_all[j:j+time_step, :, :]]
                        Y += 2 * [y_true[j:j+time_step]]
            else:
                j = self.bearing_lifetimes[i]%time_step
                while(j <= self.bearing_lifetimes[i]-time_step):
                    X.append(input_all[j:j+time_step, :, :])
                    Y.append(y_true[j:j+time_step])
                    j += time_step
        return X,Y
