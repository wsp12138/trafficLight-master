import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径
import torch
import datetime

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class Config:
    #参数的配置
    def __init__(self):

        self.algo_name='IQL'
        self.env_name='sumo'
        self.device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.seed=10
        self.gamma=0.9
        self.lr=0.001
        self.hidden_dim=128
        self.capacity=5000
        self.batch_size=64
        self.qmix_hidden_dim=64
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
