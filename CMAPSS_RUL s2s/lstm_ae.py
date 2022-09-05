import torch
import torch.nn as nn
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

@variational_estimator
class LSTM_AE(nn.Module):
    def __init__(self,input_dim,hidden_size,ouput_dim):
        super(LSTM_AE, self).__init__()
        self.encoder = BayesianLSTM(input_dim, hidden_size,prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.decoder = BayesianLSTM(hidden_size, ouput_dim,prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)

    def forward(self, x):
        x_, _ = self.encoder(x)
        x_, _ = self.decoder(x_)
        return x_

def train_model(model, train_loader, verbose, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(size_average=False)

    for epoch in range(1, epochs + 1):
        model.train()

        losses = []
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.sample_elbo(inputs=x,
                                   labels=y,
                                   criterion=criterion,
                                   sample_nbr=3) #少写一个参数

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

def quick_train(train_loader, input_dim,hidden_size, ouput_dim,verbose=True, lr=1e-3,
                epochs=15, **kwargs):#30
    model = LSTM_AE(input_dim,hidden_size,ouput_dim)
    train_model(model, train_loader, verbose, lr, epochs)

    return model.encoder, model.decoder


