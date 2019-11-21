import torch
from .torch_rnn import SimpleRNN
# from torch.autograd import grad, Variable
# import numpy as np


class FPPModel:
    def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, reg_lambda, device):

        self.state_update = state_update
        self.lr = lr
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.T = T
        self.batch_size = batch_size  # B
        self.reg_lambda = reg_lambda

        self.num_update = 1  # number of blocks to sample for each time step

        self.rnn_model = SimpleRNN(input_size, hidden_size, output_size).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()  # Input: (N, C), Target: (N)
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.rnn_model.parameters(), lr=self.lr, alpha=0.99)
        self._state = None

    def initialize_state(self):
        self._state = torch.zeros(size=[1, 1, self.hidden_size], requires_grad=True).float().to(self.device)

    def forward(self, x, y):
        """
        x: [T, 1, input_size]
        y:  [T, output_size]
        with T = 1
        """
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            state_old = self._state.clone()

            y_1, state_new = self.rnn_model(x, state_old)  # y1 = [T, 1, output_size]
            # print(y_1.size())
            y_1 = y_1[-1]
            # print(y_1.size())
            # print('--')
            correct_prediction = torch.equal(torch.argmax(y_1, 1), y)
            accuracy = correct_prediction.__float__()

            loss = self.criterion(y_1, y)
            self._state = state_new

        return loss.item(), accuracy, state_old.cpu(), state_new.cpu()

    def train(self, x_batch, s_batch, s_new_batch, y_batch):
        """
        :param
        x_batch: [T, batch_size, input_size]
        s_batch: [T, batch_size, hidden_size]
        y_batch:  [T, batch_size]

        :return
        state_old_updated: [1, batch_size, hidden_size]
        state_new_updated: [1, batch_size, hidden_size]
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # get init state and s_T
        state_old = s_batch[0:1].clone().to(self.device).detach().requires_grad_(True)
        state_new = s_new_batch[-1:].clone().to(self.device).detach().requires_grad_(True)

        output_y_batch, output_s_batch = self.rnn_model(x_batch, state_old)
        # print(x_batch.size())
        # print(s_batch.size())
        # print(output_y_batch.size())
        # print(output_s_batch[-1:].size())
        # print(state_new.size())
        loss = self.criterion(output_y_batch[-1], y_batch[-1])

        if self.state_update:
            loss += self.reg_lambda * self.mse_loss(state_new, output_s_batch[-1:])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.state_update:
            # update states: see https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html
            with torch.no_grad():
                state_old_updated = (state_old - self.lr * state_old.grad).clone().detach().cpu()
                state_new_updated = (state_new - self.lr * state_new.grad).clone().detach().cpu()

                state_old.grad.zero_()
                state_new.grad.zero_()

            # Alternative: set no_grad for state_old and state_new and manually get the gradients
            # grad_s = grad(loss, state_old, retain_graph=True)[0]
            # grad_s_t = grad(loss, state_new, retain_graph=True)[0]

            return loss.item(), state_old_updated, state_new_updated
        else:
            return loss.item(), None, None
