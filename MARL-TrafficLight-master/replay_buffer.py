import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, cfg):
        self.capacity = cfg.capacity
        self.n_agents = cfg.n_agents
        self.current_size = 0
        self.buffer = dict()
        for i in range(self.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.capacity,cfg.observation_space])
            self.buffer['u_%d' % i] = np.empty([self.capacity])
            self.buffer['r_%d' % i] = np.empty([self.capacity])
            self.buffer['o_next_%d' % i] = np.empty([self.capacity,cfg.observation_space])
        self.lock = threading.Lock()

    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.capacity:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.capacity:
            overflow = inc - (self.capacity - self.current_size)
            idx_a = np.arange(self.current_size, self.capacity)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.capacity, inc)
        self.current_size = min(self.capacity, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx


class PPOMemory:
    def __init__(self,args):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.next_states=[]
        self.dones = []
        self.batch_size = args.batch_size

    def sample(self):
        batch_step=np.arange(0,len(self.states),self.batch_size)
        indices=np.arange(len(self.states),dtype=np.int64)
        np.random.shuffle(indices)
        batches=[indices[i:i+self.batch_size] for i in batch_step]
        return np.array(self.states),np.array(self.actions),np.array(self.probs),np.array(self.rewards),np.array(self.next_states),np.array(self.dones),batches



    def push(self, state, action, probs, reward, next_state,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.next_states=[]
        self.dones = []