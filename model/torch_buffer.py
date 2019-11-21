import numpy as np
import torch
from collections import deque
import random
from heapq import heapify, heappush, heappop
from collections import namedtuple


Transition = namedtuple('Transition', ('obs', 'state_old', 'state_new', 'y'))


class ReplayBuffer():
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = deque([])
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        """
        add a transition (s_t, o_{t+1}, s_{t+1}, y_t)
        """
        if len(self._storage) < self._maxsize:
            self._storage.append(data)
            self._next_idx += 1
        else:
            self._storage.popleft()  # pop the first transition
            self._storage.append(data)

    def sample_batch(self, num_batch, batch_len):
        """
        sample M blocks of T transitions where M = num_batch and T = batch_len
        ----
        output
        ob_batch: [T, num_batch, input_size]
        state_batch: [T, num_batch, hidden_size]
        y_batch: [T, num_batch, output_size]
        idxes: a list of index
        """
        ob_batch, state_tm1_batch, state_batch, y_batch, ind_batch = [], [], [], [], []
        for m in range(num_batch):
            ob_t, state_tm1, state_t, y_t, ind_t = self.sample_successive(batch_len)
            ob_batch.append(ob_t)
            state_tm1_batch.append(state_tm1)
            state_batch.append(state_t)
            y_batch.append(y_t)
            ind_batch.append(ind_t)
            # print(ind_t)
            # print(y_t.shape)

            # print(ob_t.shape)
            # print(state_t.shape)
            # print(y_t.shape)
        ob_batch = torch.cat(ob_batch, dim=1)
        state_tm1_batch = torch.cat(state_tm1_batch, dim=1)
        state_batch = torch.cat(state_batch, dim=1)
        y_batch = torch.stack(y_batch, dim=1)
        ind_batch = np.array(ind_batch)
        return ob_batch, state_tm1_batch, state_batch, y_batch, ind_batch

    def sample_successive(self, batch_len):
        """
        sample one block of T transitions where T = batch_len
        ----
        output
        ob_batch: [T, 1, input_size]
        state_batch: [T, 1, hidden_size]
        y_batch: [T, output_size]
        idxes: a list of index
        """
        # sample index
        if len(self._storage) <= batch_len:
            idxes = [x for x in range(len(self._storage))]
        else:
            random_ix = np.random.randint(batch_len - 1, len(self._storage) - 1)
            idxes = [random_ix - x for x in range(batch_len - 1, -1, -1)]  # from random_ix-T+1 to random_ix

        # get data
        ob_batch, state_old_batch, state_new_batch, y_batch = [], [], [], []
        for i in idxes:
            obs_t, s_tm1, s_t, y_t = self._storage[i]
            ob_batch.append(obs_t)
            state_old_batch.append(s_tm1)
            state_new_batch.append(s_t)
            y_batch.append(y_t)

        ob_batch = torch.cat(ob_batch, dim=0)
        state_old_batch = torch.cat(state_old_batch, dim=0)
        state_new_batch = torch.cat(state_new_batch, dim=0)
        y_batch = torch.cat(y_batch, dim=0)
        return ob_batch, state_old_batch, state_new_batch, y_batch, idxes

    def replace_old(self, data_ind, new_data):
        """
        replace state_old in storage[data_ind] with new_data
        """
        obs, s_old, s_new, y = self._storage[data_ind]
        assert s_old.size() == new_data.size(), 'something wrong'
        self._storage[data_ind] = obs, new_data, s_new, y

    def replace_new(self, data_ind, new_data):
        """
        replace state_old in index
        """
        obs, s_old, s_new, y = self._storage[data_ind]
        assert s_old.size() == new_data.size(), 'something wrong'
        self._storage[data_ind] = obs, s_old, new_data, y

    # def _encode_sample(self, idxes):
    #     # print(ob_batch[0].shape)
    #     # print(len(ob_batch))
    #     # ob_batch = np.concatenate(ob_batch, axis=0)
    #     # ob_batch.shape = [T, 1, n_input]
    #     # state_batch.shape = [T, 1, n_unit]
    #     # y_batch.shape = [T, 1, n_class]
    #     return np.array(ob_batch), np.array(state_tm1_batch), np.array(state_batch), np.array(y_batch), idxes

    def one_sample(self):
        if len(self._storage) == 1:
            idx = 0
        else:
            idx = np.random.randint(0, len(self._storage) - 1)
        data = self._storage[idx]
        obs_t, s_tm1, s_t, y_t = data

        return obs_t, s_tm1, s_t, y_t, idx

    def check_last_two(self, idx):
        if idx <= len(self._storage) - 2:
            return True
        else:
            return False

    def not_last(self, idx):
        if idx < len(self._storage) - 1:
            return True
        else:
            return False

    def not_first(self, idx):
        if idx == 0:
            return False
        else:
            return True

    def get_sample_by_idx(self, idx):
        data = self._storage[idx]
        obs_t, s_tm1, s_t, y_t = data
        return obs_t, s_tm1, s_t, y_t, idx

    def replace_one(self, idx, data):
        self._storage[idx] = data

    def old_replace(self, idxes, obs_batch, s_tm1_batch, s_t_batch, y_batch, time_len):
        for i in range(time_len):
            obs_t = obs_batch[i]  # np.expand_dims(, axis=0)
            s_tm1 = s_tm1_batch[i]
            s_t = s_t_batch[i]
            y_t = y_batch[i]
            # print('buffer replace', obs_t.shape)
            # print('buffer replace', s_tm1.shape)
            # print('buffer replace', s_t.shape)
            # print('buffer replace', y_t.shape)
            data = (obs_t, s_tm1, s_t, y_t)
            self._storage[idxes[i]] = data

            if self.not_first(idxes[i]):
                x_t_s_j, s_tm1_s_j, s_t_s_j, y_t_s_j, idx_s_j = self.get_sample_by_idx(idxes[i] - 1)
                data = (x_t_s_j, s_tm1_s_j, s_tm1, y_t_s_j)
                self.replace_one(idx_s_j, data)
            if self.not_last(idxes[i]):
                x_t_s_j, s_tm1_s_j, s_t_s_j, y_t_s_j, idx_s_j = self.get_sample_by_idx(idxes[i] + 1)
                data = (x_t_s_j, s_t, s_t_s_j, y_t_s_j)
                self.replace_one(idx_s_j, data)

    def one_sample_n(self, j):
        if len(self._storage) == 1:
            idx = 0
        else:
            idx = self._next_idx - j - 1
        data = self._storage[idx]
        obs_t, s_tm1, s_t, s_tp1, obs_tp1, y_t = data

        return obs_t, s_tm1, s_t, y_t, idx

    def sample(self, batch_size):
        if len(self._storage) == 1:
            idxes = [0]
        else:
            idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_successive_last(self, batch_size):
        if len(self._storage) <= batch_size:
            idxes = [x for x in range(len(self._storage))]
        else:
            random_ix = len(self._storage) - 1
            idxes = [random_ix - x for x in range(batch_size - 1, -1, -1)]
        return self._encode_sample(idxes)


if __name__ == '__main__':
    pq = Replay_Buffer()
    pq['a'] = -0.1
    pq['c'] = -0.5
    pq['b'] = -0.2
    pq['d'] = -0.9
    print(pq.smallest())
    pq[pq.smallest()] = 0.0
    print(pq.smallest())
    pq[pq.smallest()] = 0.0
    print(pq.smallest())
    pq[pq.smallest()] = 0.0
    print(pq.smallest())




