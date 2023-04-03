import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from agent.qmix_net import QMixNet

class RNN(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=128):
        super(RNN, self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.GRU = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)

    def forward(self,x,h):

        x=F.relu(self.fc1(x))
        h=self.GRU(x,h)
        x=self.fc2(h)
        return x,h



class QMIX:
    def __init__(self,state_dim,action_dim,cfg,id):
        self.cfg=cfg
        self.device=cfg.device
        self.batch_size=cfg.batch_size
        self.action_dim=cfg.action_dim
        self.hidden_dim=cfg.hidden_dim
        #所有智能体共用一个网络
        self.policy_net=RNN(state_dim,action_dim).to(self.device)
        self.target_net=RNN(state_dim,action_dim).to(self.device)

        self.qmix_policy_net=QMixNet(cfg).to(cfg.device)
        self.qmix_target_net=QMixNet(cfg).to(cfg.device)
        self.eval_params=list(self.policy_net.parameters())+list(self.qmix_policy_net.parameters())
        self.optim=optim.Adam(self.eval_params,lr=cfg.lr)
        self.eval_hidden=None
        self.target_hidden=None



    def init_hidden(self):
        self.eval_hidden = torch.zeros((self.batch_size, self.hidden_dim),device=self.device)
        self.target_hidden = torch.zeros((self.batch_size, self.hidden_dim),device=self.device)


    def choose_action(self,state,hidden_state,epsilon):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values,h = self.policy_net(state,hidden_state)
        if random.random() > epsilon:
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action,h

    def update(self,transitions,update_target):
        self.init_hidden()
        self.eval_hidden=self.eval_hidden.cuda()
        self.target_hidden=self.target_hidden.cuda()
        r, o, u, o_next = [], [], [], []

        for i in range(self.cfg.n_agents):
            o.append(torch.tensor(transitions['o_%d' % i], device=self.device, dtype=torch.float))
            u.append(torch.tensor(transitions['u_%d' % i], device=self.device, dtype=torch.int64).unsqueeze(1))
            o_next.append(torch.tensor(transitions['o_next_%d' % i], device=self.device, dtype=torch.float))
            r.append(torch.tensor(transitions['r_%d' % i], device=self.device, dtype=torch.float).unsqueeze(1))

        q=[]
        next_q=[]
        for i in range(self.cfg.n_agents):
            self.init_hidden()
            q_value,self.eval_hidden=self.policy_net(o[i], self.eval_hidden)
            q.append(q_value.gather(dim=1, index=u[i]))
            next_q_value,self.target_hidden=self.target_net(o_next[i],self.target_hidden)
            next_q.append(next_q_value.max(1)[0].detach().unsqueeze(1))


        q = torch.tensor([item.cpu().detach().numpy() for item in q]).cuda()
        next_q=torch.tensor([item.cpu().detach().numpy() for item in next_q]).cuda()
        o = torch.tensor([item.cpu().detach().numpy() for item in o]).cuda()
        o_next = torch.tensor([item.cpu().detach().numpy() for item in o_next]).cuda()

        q_total = self.qmix_policy_net(q,o)
        next_q_total=self.qmix_target_net(next_q,o_next)

        r=torch.tensor([item.cpu().detach().numpy() for item in r]).cuda()
        total_r=torch.sum(r,dim=0).unsqueeze(0)
        total_r=total_r.reshape(self.cfg.batch_size,1,1)
        expect_q=total_r+self.cfg.gamma*next_q_total
        loss=nn.MSELoss()(q_total,expect_q)
        loss.requires_grad_(True)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if update_target % self.cfg.update_target == 0:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(param.data)
            for target_param,param in zip(self.qmix_target_net.parameters(),self.qmix_policy_net.parameters()):
                target_param.data.copy_(param.data)


    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'madqn_checkpoint_%d.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint_%d.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)











