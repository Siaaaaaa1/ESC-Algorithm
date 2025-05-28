import pdb
import networkx as nx
import random
import numpy as np
import logging
import copy
from queue import PriorityQueue
import sys
import heapq
import time 
import queue
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO,filename='paper_peeling4test.log',filemode='w')
logger = logging.getLogger()


def get_data(path)->list:
    data = []
    upper = []
    lower = []
    edge = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if line =="\n":
                continue
            line = line.strip('\n')
            dic = {}
            upper.append('1'+line.split(' ')[0])
            lower.append('2'+line.split(' ')[1])
            l = line.split(' ')[2:]
            for index, v in enumerate(l, start=1):
                dic.update({'weight{}'.format(index):float(v)})
            edge.append(dic)
    for i in zip(upper, lower, edge):
        data.append(i)
    return data

def build_bipartite_graph(data):
    G = nx.Graph()
    G.add_edges_from(data)
    #G.add_weighted_edges_from(data,weight='weight')
    #draw_graph(G)
    return G

        
def edge2weight(u, v, att):
    return (u, v), att

# 找到最大的符合α β要求的，包含q的顶点集
def get_alpha_beta_core_new_include_q(tmpG, q, alpha, beta):
    # 验证输入图是否是二部图
    bipartite_graph = copy.deepcopy(tmpG)
    if tmpG.number_of_edges()==0:
        return bipartite_graph
    if not nx.is_bipartite(bipartite_graph):
        raise nx.NetworkXError("The input graph is not bipartite.")

    # 初始化顶点度数
    degrees = dict(bipartite_graph.degree())

    # 初始化待移除顶点集
    to_remove = {n for n, d in degrees.items() if (n.startswith('1') and d < alpha) or (n.startswith('2') and d < beta)}

    while to_remove:
        node = to_remove.pop()
        neighbors = list(bipartite_graph.neighbors(node))
        bipartite_graph.remove_node(node)
        for neighbor in neighbors:
            degrees[neighbor] -= 1
            if (neighbor.startswith('1') and degrees[neighbor] < alpha) or (neighbor.startswith('2') and degrees[neighbor] < beta):
                to_remove.add(neighbor)
    connected_subgraphs = list(nx.connected_components(bipartite_graph))

    # 找到最大的连通子图
    # largest_connected_subgraph = max(connected_subgraphs, key=len)
    largest_connected_subgraph = []
    for i in connected_subgraphs:
        if q in i:
            largest_connected_subgraph = i

    # 创建一个子图，只包含最大连通子图的顶点
    largest_subgraph = bipartite_graph.subgraph(largest_connected_subgraph).copy()
    return largest_subgraph

def del_under_constrict(G, constrict, F=None):
    """
    删除图中不满足约束条件的边。
    
    参数:
    G -- 输入的图。
    constrict -- 约束条件，一个字典，键为维度，值为对应维度的权重阈值。
    F -- 可选参数，一个包含边的集合，这些边不应该被删除。
    
    返回:
    如果所有不满足约束条件的边都被成功删除，则返回True。
    如果由于F中的边而无法删除某条不满足约束条件的边，则返回False。
    """
    # 遍历图中的每一条边及其数据
    for edge in list(G.edges(data=True)):
        # 遍历约束条件字典
        for k, v in constrict.items():
            # 检查边的权重是否小于约束条件的阈值
            if edge[2]['weight{}'.format(k)] < v:
                # 如果F不为空，并且这条边在F中，则无法删除这条边，返回False
                if F is not None and ((edge[0],edge[1]) in F or (edge[1],edge[0]) in F):
                    return False
                # 删除不满足约束条件的边
                G.remove_edge(edge[0], edge[1])
                # 由于已经删除了一条边，跳出内层循环
                break
    # 如果所有不满足约束条件的边都被成功删除，则返回True
    return True

# 剥离peeling，目的是从小边开始搜索到所有的边，如果删除不会影响规则，则删除。
# 若单独删除边会影响规则，则查看该点是否可以被删除，以满足规则
def peeling(G, q, alpha, beta, dim=1, constrict=None, F=None):
    # 复制图G，以便在不改变原始图的情况下进行操作
    tmpG = copy.deepcopy(G)
    
    # 如果提供了限制条件constrict，则根据限制条件删除边
    if constrict is not None:
        if del_under_constrict(tmpG, constrict, F) == False:
            return None  # 如果删除失败，则返回None
        else:
            # 否则，更新tmpG为包含q的α-β核心
            tmpG = get_alpha_beta_core_new_include_q(tmpG, q, alpha, beta)
    8
    # 记录当前图的节点和边的数量
    logger.info('max include q has got {0} nodes and {1} edges'.format(tmpG.number_of_nodes(), tmpG.number_of_edges()))
    
    # 如果提供了过滤条件F，则检查图中是否包含F中的所有边
    if F is not None:
        for i in F:
            if not tmpG.has_edge(i[0], i[1]):
                return None  # 如果图中缺少F中的某条边，则返回None
    
    # 定义一个生成器函数，用于获取权重最小的边，先将边正序排列，再产生最小的边，当调用生成器时，每次迭代返回每一条边
    def get_min_edge(connectG, dim=1):
        edgelist = [i for i in connectG.edges(data=True)]  # 获取图中所有的边
        edgelist.sort(key=lambda x:x[2]['weight{}'.format(dim)])  # 按权重排序
        for edge in edgelist:
            yield edge  # 生成每一条边
    
    # 初始化变量
    my_set = []  # 用于存储被移除的边
    my_queue = []  # 用于存储待处理的节点
    max_iter = tmpG.number_of_edges()  # 获取图中边的数量
    pbar = tqdm(total=max_iter, desc='Peeling dim:{}'.format(dim))  # 创建进度条
    count = 0  # 初始化计数器
    
    # 当图中还有边时进行循环，每个边带领他的相邻节点及其节点的邻边，度数小于要求的话，将他加入到队列中
    while tmpG.number_of_edges() != 0:
        pbar.update(1)  # 更新进度条
        ### 从权重最小的边开始获取，判断边的条件，如果两侧的节点度数小于alpha或beta条件，将对应点加入队列my_queue中，等待下步处理，并且将这条边从图中移除，加入到my_set中，如果遍历到了F内的边，则直接返回
        try:
            # 获取权重最小的边
            edge = next(get_min_edge(tmpG, dim))
            e, dic = edge2weight(*edge)  # 将边的信息转换为权重
            if not tmpG.has_edge(e[0], e[1]):  # 如果图中没有这条边，则跳过
                continue
            
            # 如果F不为空，并且这条边在F中，则返回当前图
            if F is not None and ((e[0],e[1]) in F or (e[1],e[0]) in F):
                return tmpG
            
            # 移除这条边，并将其添加到my_set中
            tmpG.remove_edge(e[0], e[1])
            my_set.append(edge)
            
            # 确定边的节点两侧是1还是2，并且设置上下点为up和low
            up = e[0] if e[0].startswith('1') else e[1]
            low = e[1] if e[1].startswith('2') else e[0]
            
            # 如果节点的度数小于alpha或beta，且没有在队列中，则将其添加到队列中
            if tmpG.degree[up] < alpha and up not in my_queue:
                my_queue.append(up)
            if tmpG.degree[low] < beta and low not in my_queue:
                my_queue.append(low)
        except:
            break  # 如果出现异常，则跳出循环
        
        ### 处理队列中的节点，如果是q，则返回包含q的连通子图；
        ### 处理该节点的邻居，遍历与其邻居的所有边，按照相同的方法进行处理，为度数不足的节点扩充边
        while len(my_queue) != 0:
            u = my_queue[0]  # 获取队列的第一个节点
            my_queue.pop(0)  # 移除队列的第一个节点
            
            # 如果节点是q，则返回包含q的子图
            if u == q:
                G1 = copy.deepcopy(tmpG)
                G1.add_edges_from(my_set)
                R = list(nx.connected_components(G1))
                for i in R:
                    if q in i:
                        print('****************** note 1 *******************')
                        return G1.subgraph(i).copy()
            
            # 遍历节点的邻居
            for neighbor in list(tmpG[u]):
                edge_data = tmpG.get_edge_data(u, neighbor)  # 获取边的数据
                
                # 如果F不为空，并且这条边在F中，则返回当前图
                if F is not None and ((u, neighbor) in F or (neighbor, u) in F):
                    G1 = copy.deepcopy(tmpG)
                    G1.add_edges_from(my_set) #为什么要加入my_set,my_set不是被剥离了吗，my_set代表从tmpG中移除的边，为什么在最后输出的时候还要把my_set加进来？
                    R = list(nx.connected_components(G1))
                    for i in R:
                        if q in i:
                            print('****************** note 2 *******************')
                            return G1.subgraph(i).copy()
                
                # 移除边，并将其添加到my_set中
                tmpG.remove_edge(u, neighbor)
                my_set.append((u, neighbor, edge_data))
                
                # 如果邻居节点的度数小于alpha或beta，则将其添加到队列中
                if (neighbor.startswith('1') and tmpG.degree[neighbor] < alpha) or (neighbor.startswith('2') and tmpG.degree[neighbor] < beta):
                    my_queue.append(neighbor)
                    
                    # 如果邻居节点是q，则返回包含q的子图
                    if neighbor == q:
                        G1 = copy.deepcopy(tmpG)
                        G1.add_edges_from(my_set)
                        R = list(nx.connected_components(G1))
                        for i in R:
                            if q in i:
                                print('****************** note 3 *******************')
                                return G1.subgraph(i).copy()
        
        # 重置my_set，并增加计数器
        my_set = []
        count += 1       
    
    # 如果没有找到满足条件的子图，则返回None
    return None

#返回图中权重最小边的权重值 
def gpeel1D2f(G, dim):
    # 检查图G是否为空
    if G is None:
        # 如果图G为空，则返回-1
        return -1
    else:
        # 如果图G不为空，找到权重最小的边
        # G.edges(data=True) 返回一个包含所有边和它们属性的列表
        # key=lambda x: x[2]['weight{}'.format(dim)] 是一个排序的关键字函数，用于获取边的权重
        minedge = min(G.edges(data=True), key=lambda x: x[2]['weight{}'.format(dim)])
        # 返回找到的最小边的权重值
        return minedge[2]['weight{}'.format(dim)]

# 循环剥离2、剥离1、剥离2、剥离1
def peeling2D(G, q, alpha, beta, F=None, I=None):
    # 深拷贝图G，以便在剥皮过程中不改变原始图
    tmpG = copy.deepcopy(G)
    # 使用peeling函数进行第一次剥皮操作，获取第二个维度的最小权重
    f2 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 2, I, F), 2)
    R = []  # 初始化一个列表，用于存储每次剥皮操作后的凝聚子图
    # 如果I为空，则初始化限制条件；否则，深拷贝I并添加限制条件
    if I is None:
        constrict = {'1': 0, '2': 0}
    else:
        constrict = copy.deepcopy(I)
        constrict.update({'1': 0, '2': 0})

    # 当f2大于0时，继续剥皮操作
    while f2 > 0:
        constrict['1'] = 0
        constrict['2'] = f2
        # 使用peeling函数进行第二次剥皮操作，获取第一个维度的最小权重
        f1 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 1, constrict, F), 1)
        if f1 == -1:
            break  # 如果没有找到合适的边，则退出循环
        # 检查新找到的凝聚子图是否满足条件
        for i in R:
            if f1 <= i[0] and f2 <= i[1]:
                return R
        # 将新的凝聚子图添加到列表R中
        R.append((f1, f2))
        logger.info('当前凝聚子图：{}'.format(R))
        constrict['2'] = 0
        constrict['1'] = f1 + 0.01
        # 继续下一次剥皮操作
        f2 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 2, constrict, F), 2)

    return R

# 给出一组候选值上限，从小的开始剥离
def GetCandVal(G, alpha, beta, q, dim=3):
    # 获取包含q的alpha-beta核心子图
    tmpG = get_alpha_beta_core_new_include_q(G, q, alpha, beta)
    logger.info('max include q has got {}'.format(tmpG))
    
    Fdim = []  # 初始化一个列表，用于存储候选值
    # 定义一个内部函数，用于获取图中权重最小的边
    def get_min_edge(connectG, dim=1):
        edgelist = [i for i in connectG.edges(data=True)]
        edgelist.sort(key=lambda x: x[2]['weight{}'.format(dim)])
        for edge in edgelist:
            yield edge
    
    my_queue = []  # 初始化一个队列，用于存储待处理的节点
    max_iter = tmpG.number_of_edges()  # 获取图中边的数量
    pbar = tqdm(total=max_iter, desc='Finding F{}'.format(dim))  # 初始化进度条
    
    # 当图中边的数量不为0时，继续循环
    while tmpG.number_of_edges() != 0:
        pbar.update(1)  # 更新进度条
        try:
            # 获取权重最小的边
            edge = next(get_min_edge(tmpG, dim))
            # 将边转换为权重
            e, dic = edge2weight(*edge)
            Fdim.append(dic['weight{}'.format(dim)])  # 将权重添加到Fdim列表中
            if not tmpG.has_edge(e[0], e[1]):
                continue  # 如果边不存在，则跳过
            tmpG.remove_edge(e[0], e[1])  # 从图中移除边
            # 确定上界节点和下界节点
            up = e[0] if e[0].startswith('1') else e[1]
            low = e[1] if e[1].startswith('2') else e[0]
            # 如果上界节点的度小于alpha且不在队列中，则添加到队列
            if tmpG.degree[up] < alpha and up not in my_queue:
                my_queue.append(up)
            # 如果下界节点的度小于beta且不在队列中，则添加到队列
            if tmpG.degree[low] < beta and low not in my_queue:
                my_queue.append(low)
        except:
            break  # 如果出现异常，则跳出循环
        
        # 处理队列中的节点
        while len(my_queue) != 0:
            u = my_queue[0]
            my_queue.pop(0)
            for neighbor in list(tmpG[u]):
                tmpG.remove_edge(u, neighbor)  # 移除与u相邻的边
                # 如果邻居节点满足条件，则添加到队列
                if (neighbor.startswith('1') and tmpG.degree[neighbor] < alpha) or (neighbor.startswith('2') and tmpG.degree[neighbor] < beta):
                    my_queue.append(neighbor)
        
        # 更新tmpG为包含q的alpha-beta核心子图
        tmpG = get_alpha_beta_core_new_include_q(tmpG, q, alpha, beta)
      
    # 返回候选值列表
    return Fdim


# 定义一个名为peeling3D的函数，接受图G、参数q、alpha、beta，以及可选的F和I
# 按照F3的候选值剥离，从大的剥离到小的，先
def peeling3D(G, q, alpha, beta, F=None, I=None):
    # 深复制图G，以便在剥离过程中不改变原始图
    tmpG = copy.deepcopy(G)
    # 初始化一个字典，用于存储边的权重和对应的边
    edgedic = {}
    # 遍历图中的每一条边 edgedic 将边权重映射到边
    for edge in tmpG.edges():
        # 获取边的权重，并尝试将其添加到字典中对应的权重列表
        try:
            edgedic[tmpG.edges[edge]['weight{}'.format(3)]].append(edge)
        except KeyError:  # 如果权重不存在，则在字典中创建一个新的列表
            edgedic.update({tmpG.edges[edge]['weight{}'.format(3)]: [edge]})
    # 获取候选值列表F3，这是一个3D剥离的候选值
    F3 = GetCandVal(tmpG, alpha, beta, q, 3)
    # 初始化两个列表，R用于存储最终结果，S用于存储中间结果
    R = []
    S = []
    # 反转F3列表，以便从最大值开始剥离
    F3.reverse()
    # 记录F3的总个数
    logger.info('F3总个数{}'.format(len(F3)))
    # 如果I未提供，则初始化tI，否则深复制I并更新
    if I is None:
        tI = {'3': 0}
    else:
        tI = copy.deepcopy(I)
        tI.update({'3': 0})
    # 遍历F3列表，使用tqdm显示进度
    for f3 in tqdm(F3, desc='F3总轮次'):
        # 如果F未提供，则初始化tF为空列表，否则深复制F
        if F is None:
            tF = []
        else:
            tF = copy.deepcopy(F)
        # 获取当前f3对应的边列表
        edgeslist = edgedic[f3]
        # 遍历边列表
        for edge in edgeslist:
            # 将边添加到tF列表中
            tF.append(edge)
            # 获取边的权重1和权重2
            tmpf1 = tmpG.edges[edge]['weight{}'.format(1)]
            tmpf2 = tmpG.edges[edge]['weight{}'.format(2)]
            # 初始化一个标志变量，用于判断是否继续剥离
            ifcontinue = False
            # 检查S列表中是否有更小的值
            for prer in S:
                if tmpf1 <= prer[0] and tmpf2 <= prer[1]:
                    ifcontinue = True
                    break
            # 如果找到更小的值，则跳出循环
            if ifcontinue:
                break
            # 更新tI的值
            tI['3'] = f3
            # 调用2D剥离函数，传入当前的图、参数和边列表
            T = peeling2D(tmpG, q, alpha, beta, tF, tI)
            # 遍历2D剥离的结果
            for item in T:
                # 初始化一个标志变量，用于判断是否将结果添加到S中
                flag = False
                # 检查S列表中是否有更小的值
                for prer in S:
                    if item[0] <= prer[0] and item[1] <= prer[1]:
                        flag = True
                        break
                # 如果没有更小的值，则将结果添加到S和R中
                if flag == False:
                    S.append(item)
                    R.append(item + (f3,))
    # 返回最终结果R
    return R
                
# 定义一个名为peelingHighD的函数，接受图G、参数q、alpha、beta，维度dim，以及可选的F和I
def peelingHighD(G, q, alpha, beta, dim=4, F=None, I=None):
    # 深复制图G，以便在剥离过程中不改变原始图
    tmpG = copy.deepcopy(G)
    
    # 如果维度dim为3，则调用3D剥离函数
    if dim == 3:
        return peeling3D(tmpG, q, alpha, beta, F, I)
    
    # 初始化一个字典，用于存储边的权重和对应的边
    edgedic = {}
    # 遍历图中的每一条边
    for edge in tmpG.edges():
        # 获取边的权重，并尝试将其添加到字典中对应的权重列表
        try:
            edgedic[tmpG.edges[edge]['weight{}'.format(dim)]].append(edge)
        except KeyError:  # 如果权重不存在，则在字典中创建一个新的列表
            edgedic.update({tmpG.edges[edge]['weight{}'.format(dim)]: [edge]})
    
    # 获取候选值列表Fd，这是一个高维剥离的候选值
    Fd = GetCandVal(tmpG, alpha, beta, q, dim)
    # 初始化两个列表，R用于存储最终结果，S用于存储中间结果
    R = []
    S = []
    # 反转Fd列表，以便从最大值开始剥离
    Fd.reverse()
    # 记录Fd的总个数
    logger.info('F{}总个数{}'.format(dim, len(Fd)))
    
    # 如果I未提供，则初始化tI，否则深复制I并更新
    if I is None:
        tI = {str(dim): 0}
    else:
        tI = copy.deepcopy(I)
        tI.update({str(dim): 0})
    
    # 遍历Fd列表，使用tqdm显示进度
    for fd in tqdm(Fd, desc='F{}总轮次'.format(dim)):
        # 如果F未提供，则初始化tF为空列表，否则深复制F
        if F is None:
            tF = []
        else:
            tF = copy.deepcopy(F)
        
        # 获取当前fd对应的边列表
        edgeslist = edgedic[fd]
        # 遍历边列表
        for edge in edgeslist:
            # 将边添加到tF列表中 #QUESTION：这段是什么含义？
            tF.append(edge)
            # 获取边的权重列表，除了当前维度的权重
            tmpf_lst = []
            for idx in range(dim-1):
                tmpf_lst.append(tmpG.edges[edge]['weight{}'.format(idx+1)])
            
            # 初始化一个标志变量，用于判断是否继续剥离
            ifcontinue = False
            # 检查S列表中是否有更小的值
            for prer in S:
                if all(x <= y for x, y in zip(tmpf_lst, prer)):
                    ifcontinue = True
                    break
            # 如果找到更小的值，则跳出循环
            if ifcontinue:
                break
            
            # 更新tI的值
            tI[str(dim)] = fd
            # 递归调用剥离函数，减少一个维度
            T = peelingHighD(tmpG, q, alpha, beta, dim-1, tF, tI)
            
            # 遍历剥离的结果
            for item in T:
                # 初始化一个标志变量，用于判断是否将结果添加到S中
                flag = False
                # 检查S列表中是否有更小的值
                for prer in S:
                    if all(x <= y for x, y in zip(item, prer)):
                        flag = True
                        break
                # 如果没有更小的值，则将结果添加到S和R中
                if flag == False:
                    S.append(item)
                    R.append(item + (fd,))
    
    # 返回最终结果R
    return R



if __name__ == "__main__":
    # path = '/home/yaodi/luoxuanpu/book4dim.txt'
    # q = '123'
    # alpha, beta = 5, 2
    # path = '/home/yaodi/luoxuanpu/crime4dim.txt'
    # q = '12'
    # alpha, beta = 5, 1
    # path = '/home/yaodi/luoxuanpu/arxiv4dim.txt'
    # q = '132'
    # alpha, beta = 5, 2
    path = '/home/yaodi/luoxuanpu/crime4dim.txt'
    q = '1128'
    alpha, beta = 2, 2
    # path = './dim4graph.txt'
    # q = '12'
    # alpha, beta = 3, 2
    data = get_data(path)
    G = build_bipartite_graph(data)
    connect_subgraph = get_alpha_beta_core_new_include_q(G, q, alpha, beta)
    origG = G.subgraph(connect_subgraph).copy()
    print(len(origG.edges()))
    # nx.set_edge_attributes(G, 0, "visited")
    
    # lst = [(285.9, 67.3, 137.7, 738.8), (369.7, 449.6, 57.1, 738.8), (390.8, 173.8, 15.9, 682.4), (285.9, 67.3, 551.9, 549.5), (544.2, 604.1, 137.7, 549.5), (612.5, 85.3, 120.9, 453.9), (544.2, 48.7, 694.7, 317.0), (686.4, 48.7, 137.7, 317.0), (503.3, 69.7, 160.2, 51.3), (185.6, 308.9, 204.4, 11.2)]
    # lst = [(544.2, 48.7, 694.7), (285.9, 67.3, 551.9), (185.6, 308.9, 204.4), (503.3, 69.7, 160.2), (544.2, 604.1, 137.7), (686.4, 48.7, 137.7), (612.5, 85.3, 120.9)]
    lst = [(544.2, 604.1), (612.5, 85.3), (686.4, 48.7)]
    # lst = [(686.4,)]
    # lst = [(0, 0, 0, 738.8)]
    el = []
    for i in lst:
        for e in origG.edges(data=True):
            f = True
            for idx, v in enumerate(i):
                if e[2]['weight{}'.format(idx+1)] < v:
                    f = False
                    break
            if f == True:
                el.append(e)
        ag = nx.Graph()
        ag.add_edges_from(el)
        ag1 = get_alpha_beta_core_new_include_q(ag, q, alpha, beta)
        print(len(ag1.edges()))
    
    
    starttime = time.time()
    # print(GetCandVal(origG, alpha, beta, q, 3))
    # print(get_alpha_beta_core_new_include_q(G,q,alpha,beta))
    # pdb.set_trace()
    print(gpeel1D2f(peeling(origG, q, alpha, beta, 4),4))
    # print(expand(origG, alpha, beta, q, 1))
    # print(peeling2D(origG, q, alpha, beta))
    # res = peelingHighD(origG, q, alpha, beta, 3)
    # print(res)
    # logger.info(res)
    endtime = time.time()
    print(endtime - starttime)
    logger.info(endtime - starttime)