



import os
def save_data(path,data,filename):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename + '_data.txt'), "w") as file:
        for value in data:
            file.write("%s\n" % value)


import numpy as np
eps_b=0.05 # winner更新权重
eps_n=0.0005 # 邻居节点更新权重
lw = 1000
opw = 10
def get_new_position(winnerpos, nodepos):  # TODO
    """
    :param winnerpos: winner位置
    :param nodepos: 输入信号位置
    :return: winner更新后位置
    """
    move_delta = [eps_b * (nodepos[0] - winnerpos[0]), eps_b * (nodepos[1] - winnerpos[1])]
    newpos = [winnerpos[0] + move_delta[0], winnerpos[1] + move_delta[1]]
    return newpos

def get_new_position_neighbors(neighborpos, nodepos): #TODO
    """

    :param neighborpos: 邻居节点位置
    :param nodepos: 输入信号位置
    :return: 邻居节点更新后位置
    """
    movement = [eps_n * (nodepos[0] - neighborpos[0]), eps_n * (nodepos[1] - neighborpos[1])]
    newpos = [neighborpos[0] + movement[0], neighborpos[1] + movement[1]]
    return newpos

def get_average_dist(a, b):
    """
    插入新节点位置
    :param a:
    :param b:
    :return: 返回中间位置
    """
    av_dist = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
    return av_dist



def distance(a, b):
    """
    计算距离
    输入节点坐标
    :param a:
    :param b:
    :return: 距离的平方
    """

    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)

# def find_near(a,b):

#
#     wx = a[:2]
#     wy = b[:2]
#     opx = a[2:]
#     opy = b[2:]
#     KL1 = distance(wx,wy)
#     KL2 = 0
#     for x, y in zip(opx, opy):
#         KL2 += abs(x - y)
#     return KL1+np.exp(KL2)

def find_near(a,b):

    wx = a[:2]
    wy = b[:2]
    opx = a[2:]
    opy = b[2:]
    KL1 = distance(wx,wy)
    opx=np.array(opx)
    opy=np.array(opy)
    dist=np.linalg.norm(opx-opy)
    return KL1+dist