
import os.path
import sys
import numpy as np
import networkx as nx
import imageio
import re
import glob
import operator
from past.builtins import xrange
import matplotlib.pyplot as plt
from future.utils import iteritems
from utils import *
# 聚类跟初始节点有关，还跟超参数有关






class GNG():

    def __init__(self,data,map_origin,G,pos_origin,eps_b=0.05,eps_n=0.0005,max_age=25,
                 lambda_=100,alpha=0.5,d=0.0005,max_nodes=13):

        self.map_origin=map_origin
        self.G=G
        self.pos_origin=pos_origin
        self.graph = nx.Graph()
        self.data = data
        self.eps_b = eps_b # winner更新权重
        self.eps_n = eps_n # 邻居节点更新权重
        self.max_age = max_age # 最大边年龄
        self.lambda_ = lambda_ # 插入节点迭代次数
        self.alpha = alpha # 减小最大误差和其邻居最大误差权重
        self.d = d  # 全局误差减小权重
        self.max_nodes = max_nodes # 最大节点数
        self.pos = None
        self.netmap = {}  # 生成网络字典
        # 从原始数据中随机抽取两个最为初始信号
        random_num1 = np.random.randint(0, 14)
        random_num2 = np.random.randint(0, 14)
        node1 = self.map_origin[random_num1]
        node2 = self.map_origin[random_num2]

        while node1 == node2:
            random_num1 = np.random.randint(0, 14)
            node1 = self.map_origin[random_num1]
            if node1 != node2:
                break


        self.count = 0
        self.graph.add_node(self.count, pos=(node1[0], node1[1]), error=0)
        self.netmap.update({self.count: node1})
        self.count += 1
        self.graph.add_node(self.count, pos=(node2[0], node2[1]), error=0)
        self.netmap.update({self.count: node2})
        self.count += 1


    def determine_2closest_vertices(self, cursignal):
        """
        :param cursignal: 当前节点坐标
        :return: 最近节点和次近节点的节点编号和距离
        """


        self.pos = nx.get_node_attributes(self.graph, 'pos')
        templist = []
        #c是下标 s是netpos里面存储的坐标
        for c, s in iteritems(self.netmap):

            #print(cursignal==s)
            # KL = scipy.stats.entropy(cursignal,s)
            KL = find_near(cursignal, s)
            templist.append([c, KL])
        KLlist = np.array(templist)
        ind = np.lexsort((KLlist[:, 0], KLlist[:, 1]))
        KLlist = KLlist[ind]

        return KLlist[0], KLlist[1]



    def update_winner(self, cursignal):
        """
        更新winner节点
        :param cursignal:输入信号位置
        :return:
        """

        winner1, winner2 = self.determine_2closest_vertices(cursignal) # 根据输入信号找到最近的两个节点
        winnernode = winner1[0] # 最近节点编号
        winnernode2 = winner2[0] # 次近节点编号
        win_dist_from_node = winner1[1] # 最近节点距离

        #根据输入信号得到点与输入信号最近的节点之间的差距error
        errorvectors = nx.get_node_attributes(self.graph, 'error')

        #根据图里的节点拿到误差
        error1 = errorvectors[winnernode]
        newerror = error1 + win_dist_from_node
        self.graph.add_node(winnernode, error=newerror) # 更新最近节点误差
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        newposition = get_new_position(self.pos[winnernode], cursignal)
        self.graph.add_node(winnernode, pos=newposition) # 更新最近节点位置

        neighbors = nx.all_neighbors(self.graph, winnernode)
        age_of_edges = nx.get_edge_attributes(self.graph, 'age')

        for n in neighbors:
            newposition = get_new_position_neighbors(self.pos[n], cursignal)
            self.graph.add_node(n, pos=newposition)
            key = (winnernode, n)

            # 与winner连接的边年龄+1
            if key in age_of_edges:
                newage = age_of_edges[(winnernode, n)] + 1

            else:
                newage = age_of_edges[(n, winnernode)] + 1

            self.graph.add_edge(winnernode, n, age=newage)

        # 最近和次近无边则添加边，有边边年龄置为0
        if self.graph.get_edge_data(winnernode, winnernode2) is not None:
            self.graph.add_edge(winnernode, winnernode2, age=0)

        else:
            self.graph.add_edge(winnernode, winnernode2, age=0)

        age_of_edges = nx.get_edge_attributes(self.graph, 'age')
        for edge, age in iteritems(age_of_edges):

            if age > self.max_age:  # 删除超龄边
                self.graph.remove_edge(edge[0], edge[1])

        for node in list(self.graph.nodes()): # 删除孤立节点
            if not list(self.graph.neighbors(node)):
                self.graph.remove_node(node)
                self.netmap.pop(node)


    def save_img(self,fignum, output_images_dir):
        fig = plt.figure(fignum)  # 创建画布
        ax = fig.add_subplot(111)

        nx.draw(self.G, self.pos_origin, with_labels=False, node_size=100, width=1.5)
        position = nx.get_node_attributes(self.graph, 'pos')

        nx.draw(self.graph, position, node_color='r', node_size=100, with_labels=False, edge_color='g', width=1.5)

        plt.title('Growing Neural Gas')
        plt.savefig("{0}/{1}.png".format(output_images_dir, str(fignum)))

        plt.clf()  # 清空图形，保持窗口打开
        plt.close(fignum)


    def run(self,max_iterations, output_images_dir='images_new'):
        """
        节点更新，插入新节点
        :param max_iterations:
        :param output_images_dir:
        :return:
        """
        if not os.path.isdir(output_images_dir):
            os.makedirs(output_images_dir)

        #print('Output images will be saved in: {0}'.format(output_images_dir))

        fignum = 0
        #保存初始化的图 两个红色点为随机选中的两个初始化的网络
        #self.save_img(fignum,output_images_dir)

        for i in xrange(1, max_iterations):
            #print("Iterating..{0}/{1}".format(i,max_iterations))

            for x in self.data: # 遍历数据
                #
                self.update_winner(x)

                if i % self.lambda_ == 0 and len(self.graph.nodes()) <= self.max_nodes:  # i等于迭代次数并且小于最大节点数插入新节点

                    errorvectors = nx.get_node_attributes(self.graph,'error')

                    node_largest_error = max(iteritems(errorvectors), key=operator.itemgetter(1))[0]  # 获得最大误差节点

                    neighbors = self.graph.neighbors(node_largest_error)

                    max_error_neighbor = None
                    max_error = -1

                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    for n in neighbors:  # 寻找邻居节点中的最大误差节点
                        if errorvectors[n] > max_error:
                            max_error = errorvectors[n]
                            max_error_neighbor = n

                    self.pos = nx.get_node_attributes(self.graph, 'pos')

                    newnodepos = get_average_dist(self.pos[node_largest_error],self.pos[max_error_neighbor]) # 插入新节点位置

                    newnode = self.count
                    self.count = self.count + 1
                    newsignal_ = self.netmap[node_largest_error][2:]
                    newsignal = np.hstack((newnodepos, newsignal_))
                    self.netmap.update({newnode: newsignal})

                    self.graph.add_node(newnode, pos=newnodepos) # 添加新节点

                    self.graph.add_edge(newnode, max_error_neighbor, age=0)  # 连接新节点和邻居最大误差节点
                    self.graph.add_edge(newnode, node_largest_error, age=0)  # 连接新节点和最大误差节点

                    self.graph.remove_edge(max_error_neighbor, node_largest_error) # 断开最大误差节点和其邻居最大误差的边

                    errorvectors = nx.get_node_attributes(self.graph, 'error')

                    error_max_node = self.alpha * errorvectors[node_largest_error] # 更新最大误差节点误差和其邻居最大节点误差
                    error_max_second = self.alpha * max_error
                    self.graph.add_node(max_error_neighbor, error = error_max_second)
                    self.graph.add_node(node_largest_error, error = error_max_node)
                    self.graph.add_node(newnode, error = error_max_node) # 更新后的最大误差节点的误差赋值给新节点
                    fignum += 1
                    #self.save_img(fignum, output_images_dir)

                errorvectors = nx.get_node_attributes(self.graph, 'error')

                for i in self.graph.nodes():
                    olderror = errorvectors[i]
                    newerror = olderror - self.d * olderror
                    self.graph.add_node(i, error = newerror)

    def number_of_clusters(self):
        """

        :return: 返回连通图数量
        """
        noc = nx.number_connected_components(self.graph)
        return noc

    def cluster_data(self):
        """
        对输入信号进行聚类
        :return:
        """
        unit_to_cluster = np.zeros(self.count) # 定义全零数组
        cluster = 0
        # 对gng节点进行聚类
        for c in nx.connected_components(self.graph):
            for unit in c:
                unit_to_cluster[int(unit)] = cluster
            cluster += 1

        clustered_data = []
        for obs in self.data:
            nearest_units = self.determine_2closest_vertices(obs) # 找离输入信号最近的网络节点
            s = nearest_units[0][0]
            clustered_data.append((obs,unit_to_cluster[int(s)]))

        return clustered_data

    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.graph)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('clusters.png')




def sort_nicely(limages):

    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)',key)]

    limages = sorted(limages, key=alphanum_key)
    return limages

def convert_image_to_gif(output_images_dir, output_gif):

    image_dir = "{0}/*.png".format(output_images_dir)
    list_images = glob.glob(image_dir)
    file_names = sort_nicely(list_images)
    images = [imageio.imread(fn) for fn in file_names]
    imageio.mimsave(output_gif, images)


def get_gng_result(signal):
    map = {}
    count = 0
    G = nx.Graph()

    for i in signal:
        G.add_node(count, pos=(i[0], i[1]))
        map.update({count: i})
        count += 1
    pos = nx.get_node_attributes(G, 'pos')

    gng = GNG(signal,map,G,pos)
    if gng is not None:
        gng.run(max_iterations=3500, output_images_dir='gng1.0')
        return gng.cluster_data()
