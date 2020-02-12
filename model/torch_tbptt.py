import torch
from .torch_rnn import SimpleRNN
# from torch.autograd import grad, Variable
# import numpy as np


class TBPTTModel:
    def __init__(self, input_size, hidden_size, output_size, lr, state_update, batch_size, T, overlap, device):

        self.state_update = state_update
        self.lr = lr
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.T = T
        self.batch_size = batch_size  # B
        self.overlap = overlap

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
        y:  [T, 1]
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
        state_old = s_batch[0:1].clone().to(self.device).detach()
        output_y_batch, output_s_batch = self.rnn_model(x_batch, state_old)
        # Question: update all outputs or just one?
        if self.overlap:
            # if overlap, only computed the loss for the final prediction
            loss = self.criterion(output_y_batch[-1], y_batch[-1])
        else:
            # if non-overlap, computed the loss for all predictions
            output_y_batch = output_y_batch.squeeze(dim=1)
            y_batch = y_batch.squeeze(dim=1)
            loss = self.criterion(output_y_batch, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), None, None
