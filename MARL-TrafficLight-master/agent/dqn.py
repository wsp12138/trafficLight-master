import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=128):
        super(MLP, self).__init__()
        #三层全连接层
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        #激活层
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim, action_dim, cfg, id):

        self.cfg=cfg
        self.action_dim = action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.policy_net = MLP(state_dim, action_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器
        self.agent_id = id

    def choose_action(self, state,epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self, transitions,update_target):
        r = transitions['r_%d' % self.agent_id]
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]
        o = torch.tensor(o, device=self.device, dtype=torch.float)
        u = torch.tensor(u, device=self.device, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, device=self.device, dtype=torch.float)
        o_next = torch.tensor(o_next, device=self.device, dtype=torch.float)
        q_values = self.policy_net(o).gather(dim=1, index=u)

        next_q_values = self.target_net(o_next).max(1)[0].detach()
        expected_q_values = r + self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if update_target%self.cfg.update_target==0:
            for target_param, param in zip(self.target_net.parameters(),
                                           self.policy_net.parameters()):
                target_param.data.copy_(param.data)



    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint_%d.pth' % self.agent_id)

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint_%d.pth' % self.agent_id))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)