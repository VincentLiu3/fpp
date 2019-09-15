import numpy as np
from collections import deque
import random

from heapq import heapify, heappush, heappop

class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """
    
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).
        
        super(priority_dict, self).__setitem__(key, val)
        
        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        
        while self:
            yield self.pop_smallest()

class Replay_Buffer():
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
        if len(self._storage) < self._maxsize:
            self._storage.append(data)
            self._next_idx += 1
        else:
            self._storage.popleft()
            self._storage.append(data)

    def replace_one(self, idx, data):
        self._storage[idx] = data

    def _encode_sample(self, idxes):
        obses_t, ses_tm1, ses_t, ys_t = [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, s_tm1,s_t, y_t = data
            obses_t.append(np.array(obs_t))
            ses_tm1.append(np.array(s_tm1))
            ses_t.append(np.array(s_t))
            ys_t.append(np.array(y_t))
        return np.array(obses_t), np.array(ses_tm1), np.array(ses_t), np.array(ys_t), idxes

    def one_sample(self):
        if len(self._storage) == 1:
            idx = 0
        else:
            idx = np.random.randint(0, len(self._storage) - 1)
        data = self._storage[idx]
        obs_t, s_tm1, s_t, y_t = data
        
        return obs_t, s_tm1, s_t, y_t, idx

    def check_last_two(self,idx):
        if idx <= len(self._storage)-2:
            return True
        else:
            return False

    def check_last(self,idx):
        if idx < len(self._storage)-1:
            return True
        else:
            return False

    def check_first(self,idx):
        if idx == 0:
            return False
        else:
            return True

    def get_sample_by_idx(self,idx):
        data = self._storage[idx]
        obs_t, s_tm1, s_t, y_t = data
        return obs_t, s_tm1, s_t, y_t, idx


    def one_sample_n(self,j):
        if len(self._storage) == 1:
            idx = 0
        else:
            idx = self._next_idx-j-1
        data = self._storage[idx]
        obs_t, s_tm1, s_t, s_tp1, obs_tp1, y_t = data
        
        return obs_t, s_tm1, s_t, y_t, idx

    def replace(self, idxes, obs_t, s_tm1, s_t, y_t, batch_size):
        for i in range(batch_size):
            data = (obs_t[i], s_tm1[i], s_t[i], y_t[i])
            self._storage[idxes[i]] = data
            if self.check_first(idxes[i]) == True:
                x_t_s_j, s_tm1_s_j,s_t_s_j, y_t_s_j,idx_s_j = self.get_sample_by_idx(idxes[i]-1)
                data = (x_t_s_j, s_tm1_s_j,s_tm1[i], y_t_s_j)
                self.replace_one(idx_s_j, data)
            if self.check_last(idxes[i]) == True:
                x_t_s_j, s_tm1_s_j,s_t_s_j, y_t_s_j,idx_s_j = self.get_sample_by_idx(idxes[i]+1)
                data = (x_t_s_j, s_t[i],s_t_s_j, y_t_s_j)
                self.replace_one(idx_s_j, data)
    
    def sample(self, batch_size):
        if len(self._storage) == 1:
            idxes = [0]
        else:
            idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_successive(self, batch_size):
        if len(self._storage) <= batch_size:
            idxes = [x for x in range(len(self._storage))]
        else:
            random_ix = np.random.randint(batch_size-1, len(self._storage) - 1)
            idxes = [random_ix-x for x in range(batch_size-1,-1,-1)]
        return self._encode_sample(idxes)

    def sample_successive_last(self, batch_size):
        if len(self._storage) <= batch_size:
            idxes = [x for x in range(len(self._storage))]
        else:
            random_ix = len(self._storage) - 1
            idxes = [random_ix-x for x in range(batch_size-1,-1,-1)]
        return self._encode_sample(idxes)


class Prioritized_Replay_Buffer():
    def __init__(self, size, alpha):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        random.seed(0)
        self._storage = deque([])
        self._maxsize = size
        self._next_idx = 0
        self.pq = priority_dict()
        self.alpha = alpha

    def __len__(self):
        return len(self._storage)

    def add(self, data, priority):
        if len(self._storage) < self._maxsize:
            self._storage.append(data)
            self.pq[self._next_idx] = -priority
            self._next_idx = (self._next_idx+1)%self._maxsize
        else:
            self._storage[self._next_idx] = data
            self.pq[self._next_idx] = -priority
            self._next_idx = (self._next_idx+1)%self._maxsize

    def replace_one(self, idx, data):
        self._storage[idx] = data

    def _encode_sample(self, idxes):
        obses_t, ses_tm1, ses_tp1, obses_tp1, ys_t = [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, s_tm1, s_tp1, obs_tp1, y_t = data
            obses_t.append(np.array(obs_t))
            ses_tm1.append(np.array(s_tm1))
            ses_tp1.append(np.array(s_tp1))
            obses_tp1.append(np.array(obs_tp1))
            ys_t.append(np.array(y_t))
        return np.array(obses_t), np.array(ses_tm1), np.array(ses_tp1), np.array(obses_tp1), np.array(ys_t), idxes

    def one_sample(self):
        if len(self._storage) == 1:
            idx = 0
        else:
            idx = np.random.randint(0, len(self._storage) - 1)
        data = self._storage[idx]
        obs_t, s_tm1, s_tp1, obs_tp1, y_t = data
        
        return obs_t, s_tm1, s_tp1, obs_tp1, y_t, idx

    def check_last(self,idx):
        if idx >= len(self._storage)-2:
            return True
        else:
            return False

    def get_sample_by_idx(self,idx):
        data = self._storage[idx]
        obs_t, s_tm1, s_tp1, obs_tp1, y_t = data
        return obs_t, s_tm1, s_tp1, obs_tp1, y_t, idx


    def one_sample_n(self,j):
        if len(self._storage) == 1:
            idx = 0
        else:
            idx = self._next_idx-j-1
        data = self._storage[idx]
        obs_t, s_tm1, s_tp1, obs_tp1, y_t = data
        
        return obs_t, s_tm1, s_tp1, obs_tp1, y_t, idx

    def replace(self, idxes, obs_t, s_tm1, s_tp1, obs_tp1, y_t, batch_size):
        for i in range(batch_size):
            data = (obs_t[i], s_tm1[i], s_tp1[i], obs_tp1[i], y_t[i])
            self._storage[idxes[i]] = data
            if self.check_last(idxes[i]) == False:
                x_t_s_j, s_tm1_s_j, s_tp1_s_j, x_tp1_s_j, y_t_s_j,idx_s_j = self.get_sample_by_idx(idxes[i]+2)
                data = (x_t_s_j, s_tp1[i], s_tp1_s_j, x_tp1_s_j, y_t_s_j)
                self.replace_one(idx_s_j, data)
    
    def sample(self, batch_size):
        if len(self._storage) == 1:
            idxes = [0]
        else:
            idxes = []
            for _ in range(batch_size):
                if random.random() >= self.alpha:
                    idxes.append(np.random.randint(0, len(self._storage) - 1))
                else:
                    idx = self.pq.smallest()
                    self.pq[idx] /= 2 
                    idxes.append(idx)
            # idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_successive(self, batch_size):
        if len(self._storage) <= batch_size:
            idxes = [x for x in range(len(self._storage))]
        else:
            random_ix = np.random.randint(batch_size-1, len(self._storage) - 1)
            idxes = [random_ix-x for x in range(batch_size-1,-1,-1)]
        return self._encode_sample(idxes)

    def sample_successive_last(self, batch_size):
        if len(self._storage) <= batch_size:
            idxes = [x for x in range(len(self._storage))]
        else:
            random_ix = len(self._storage) - 1
            idxes = [random_ix-x for x in range(batch_size-1,-1,-1)]
        return self._encode_sample(idxes)


if __name__=='__main__':
    pq = priority_dict()
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




