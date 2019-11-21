import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.RNN(input_size, hidden_size, 1, batch_first=False)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        """
        :param
        x: (T, batch_size, input_size)
        state: (1, batch_size, hidden_size)

        :return
        y: (T, batch_size, output_size)
        state: (1, batch_size, hidden_size)
        """
        output, state = self.rnn_layer(x, state)  # output = (T, batch_size, hidden_size)
        # print(output.size())
        # print(state.size())
        y = self.output_layer(output)
        # print(y.size())
        # print('---')
        return y, state

    # def make_module_parameter(self):
    #     """
    #     torch.Size([16, 2])
    #     torch.Size([16, 16])
    #     torch.Size([16])
    #     torch.Size([16])
    #
    #     torch.Size([2, 16])
    #     torch.Size([2])
    #     """
    #     params = list(p for p in self.parameters() if p.requires_grad)
    #     # single_param = MergedVariable.join(params, requires_grad=True, as_leaf=True)
    #     # new_params = single_param.cleave(share_data = True)
    #     # set_module_parameters(module, (p for p in new_params), only_requiring_grad = only_requiring_grad)
    #     # assert len(list(module.parameters()))==0
    #     return params


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        print('using GRU')
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        """
        x: (1, 1. input_size)
        state: (1, hidden_size)
        output: (1, output_size)
        """
        output, state = self.rnn_layer(x, state)
        output = self.output_layer(output.squeeze(1))
        return output, state