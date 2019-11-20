import torch
from torch.autograd import grad, Variable
from .torch_rnn import SimpleRNN, SimpleLSTM
import numpy as np
# from artemis.general.nested_structures import nested_map
# from uoro_demo.torch_utils.interfaces import TrainableStatefulModule
# from uoro_demo.torch_utils.torch_helpers import clone_em
# from uoro_demo.torch_utils.training import set_optimizer_learning_rate
# from uoro_demo.torch_utils.variable_workshop import MergedVariable

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')


class Param2Vec():
    def __init__(self, model):
        """
        get a list of trainable variables
        """
        self.param_list = []
        # self.state_param_list = []
        self.size_list = []
        for p in model.parameters():
            if p.requires_grad:
                self.param_list.append(p)
                self.size_list.append(p.size())

        # for p in model.rnn_layer.parameters():
        #     if p.requires_grad:
        #         self.state_param_list.append(p)

        self.num_list = len(self.param_list)

    def merge(self, var_list):
        """
        merge a list of variables to a vector
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            var = var_list[i]
            if var is not None:
                theta_list.append(var.flatten())
            else:
                theta_list.append(torch.zeros(self.size_list[i]).flatten())
        return torch.cat(theta_list)

    def split(self, var_vec):
        """
        split a vec to a list
        """
        var_list = []
        count = 0
        for i in range(len(self.size_list)):
            prod_size = np.prod(self.size_list[i])
            var_list.append(var_vec[count:(count+prod_size)].reshape(self.size_list[i]))
            count += prod_size
        return var_list


class UORO_Model():
    def __init__(self, input_size, hidden_size, output_size, lr, optimizer_type='Adam',
                 epsilon_perturbation=1e-7, epsilon_stability=1e-7):
        """
        """
        # super(UORO_Model, self).__init__()
        assert optimizer_type in ['SGD', 'Adam']
        self.hidden_size = hidden_size
        self.rnn_model = SimpleRNN(input_size, hidden_size, output_size)

        self.epsilon_perturbation = epsilon_perturbation
        self.epsilon_stability = epsilon_stability
        self.criterion = torch.nn.CrossEntropyLoss()
        if optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.rnn_model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.rnn_model.parameters(), lr=lr)

        # lambda1 = lambda epoch: 1 / (1 + 0.003 * np.sqrt(epoch))
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=0.99)

        self.nn_param = Param2Vec(self.rnn_model)
        self.s_tilda = None
        self.theta_tilda = None
        self._state = None

    def initialize_state(self):
        self._state = torch.zeros(size=[1, 1, self.hidden_size], requires_grad=True).float()

    def forward(self, x, y):
        """
        x: input of size (1, 1, input_size)
        s: previous recurrent state of size (1, 1, hidden_size)
        y: target of size (1,)

        self.s_toupee: column vector of size (state, )
        self.theta_toupee: row vector of size (params, )
        """
        # print('---')
        # print(x.size())
        # print(s.size())
        # print(y.size())
        # state_old = torch.tensor(self._state, requires_grad=True)
        state_old = self._state.clone().detach().requires_grad_(True)

        y_1, state_new = self.rnn_model(x, state_old)
        loss = self.criterion(y_1, y)

        correct_prediction = torch.equal(torch.argmax(y_1, 1), y)  #torch.argmax(y, 1)
        accuracy = correct_prediction.__float__()

        delta_s = grad(loss, state_old, retain_graph=True)[0]
        delta_theta = grad(loss, self.nn_param.param_list, retain_graph=True)
        delta_theta_vec = self.nn_param.merge(delta_theta)

        if self.s_tilda is None or self.theta_tilda is None:
            self.s_tilda = Variable(torch.zeros(*state_old.squeeze().size()))  # (batch_size, state_dim)
            self.theta_tilda = Variable(torch.zeros(*delta_theta_vec.size()))  # (n_params, )

        # print(self.s_tilda.size())
        # print(self.theta_tilda.size())
        g_t1 = torch.dot(delta_s.squeeze(), self.s_tilda) * self.theta_tilda + delta_theta_vec
        g_t1_list = self.nn_param.split(g_t1)

        # ForwardDiff
        state_old_perturbed = state_old + self.s_tilda * self.epsilon_perturbation  #.detach()
        state_new_perturbed = self.rnn_model(x, state_old_perturbed)[1]
        s_forwarddiff = (state_new_perturbed - state_new)/self.epsilon_perturbation

        # Backprop
        nu_vec = Variable(torch.round(torch.rand(*state_old.size())) * 2 - 1)
        delta_theta_g = grad(outputs=state_new, inputs=self.nn_param.param_list, grad_outputs=nu_vec, allow_unused=True,
                             retain_graph=True)
        delta_theta_g_vec = self.nn_param.merge(delta_theta_g)

        rho_0 = torch.sqrt(self.theta_tilda.norm()/(s_forwarddiff.norm() + self.epsilon_stability)) + self.epsilon_stability
        rho_1 = torch.sqrt(delta_theta_g_vec.norm()/(nu_vec.norm() + self.epsilon_stability)) + self.epsilon_stability

        self.s_tilda = (rho_0 * s_forwarddiff.squeeze() + rho_1 * nu_vec.squeeze()).detach()
        self.theta_tilda = (self.theta_tilda / rho_0 + delta_theta_g_vec / rho_1).detach()

        # self.optimizer.zero_grad()
        for i in range(len(self.nn_param.param_list)):
            self.nn_param.param_list[i].grad = g_t1_list[i]
        self.optimizer.step()
        # self.scheduler.step()

        self._state = state_new

        return loss.item(), accuracy, state_old, state_new




