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
from paper_debug import get_data, get_alpha_beta_core_new_include_q, build_bipartite_graph, peeling, edge2weight, gpeel1D2f

logging.basicConfig(level=logging.INFO,filename='paper_peeling4test.log',filemode='w')
logger = logging.getLogger()

def community2skylineValue(G, all_dim = 1):
    """
    convert the community to skyline value,for example G->(f1,f2,...)
    all_dim: the number of the attributes.
    """
    if G is None:
        return []
    edgelist = list(G.edges)
    res = []
    for dim in range(all_dim):
        edgelist.sort(key=lambda x:G.edges[x]['weight{}'.format(dim+1)])
        res.append(G.edges[edgelist[0]]['weight{}'.format(dim+1)])
    return res

### 如果需要expand 理应从q搜索所有边中权重最大的，添加到其中，先扩展再剥离
### 扩展：首先获取q点周围权重最大的边，再继续扩展，直到可以找到αβ子图
def expand(G, q, alpha, beta, dim=1):
    # 复制原始图G，以便在不改变原始图的情况下进行操作
    tmpG = copy.deepcopy(G)
    print('max include q has got', tmpG)
    
    # 定义一个内部函数，用于获取权重最大的边的列表，并按权重从大到小排序
    def get_max_edge(connectG, dim=1):
        edgelist = [i for i in connectG.edges(data=True)]  # 获取图中所有的边
        edgelist.sort(key=lambda x:x[2]['weight{}'.format(dim)], reverse=True)  # 按权重排序 {}是占位符，将dim放入占位符中
        return edgelist
    
    # 定义一个内部函数，用于检查是否满足引理1的条件 
    def lemma1(connectG):
        return alpha * beta - alpha - beta <= connectG.number_of_edges() - connectG.number_of_nodes()
    
    # 定义一个内部函数，用于检查是否满足引理2的条件
    def lemma2(connectG):
        number_upper, number_lower = 0, 0
        for item in list(connectG.nodes):
            if item.startswith('1') and connectG.degree(item) >= alpha:  # 节点以'1'开头且度数大于等于alpha
                number_upper += 1
            elif item.startswith('2') and connectG.degree(item) >= beta:  # 节点以'2'开头且度数大于等于beta
                number_lower += 1
        return number_upper >= beta and number_lower >= alpha  # 检查上界和下界节点的数量
    
    G1 = nx.Graph()  # 创建一个新的空图G1
    # 按照边权从大到小排序，只考虑与节点q相连的边
    edges_contain_q = [(u, v, data) for u, v, data in tmpG.edges(data=True) if q in (u, v)]
    print(edges_contain_q)
    node_edges = sorted(edges_contain_q, key=lambda x: x[2]['weight{}'.format(dim)], reverse=True)
    # 根据q的类型确定阈值limt
    limt = alpha if q.startswith('1') else beta
    # 找到权重值大于limt的边权重   [0]s [1]e     [2]v  5 3 3 3  1  a=3
    print(node_edges)
    if len(node_edges)==0:
        th_largest_value = 0
    if len(node_edges) >= limt and len(node_edges) != 0:
        th_largest_value = node_edges[limt-1][2]['weight{}'.format(dim)]
    else:
        th_largest_value = node_edges[-1][2]['weight{}'.format(dim)]
    print('find q th val', th_largest_value)
    
    # 找到所有权重大于阈值的边，并添加到G1中，同时从tmpG中移除
    edgelist = [i for i in list(tmpG.edges(data=True)) if i[2]['weight{}'.format(dim)] > th_largest_value]
    G1.add_edges_from(edgelist)
    tmpG.remove_edges_from(edgelist)
    print(G1.number_of_edges())
    print(tmpG.number_of_edges())
    print(G1.has_node(q))
    
    pre_len = 0  # 初始化之前的长度
    edgelist = get_max_edge(tmpG, dim)  # 获取tmpG中权重最大的边的列表
    index = 0  # 初始化索引
    max_iter = tmpG.number_of_edges()  # 获取tmpG中边的数量
    pbar = tqdm(total=max_iter, desc='expanding')  # 创建进度条
    # 按照权重从大到小的顺序获取边，每次添加权重最大的边，组成新的图，并找到其中最大包含q的连通子图
    # 由q点作为起始扩展，包括q点周围所有权重大的边，（QUESTION？猜想，如果略过一个会导致q点周围边更小的呢？效果是什么？ 定一个规则）
    while tmpG.number_of_edges() != 0:  # 当tmpG中还有边时
        pbar.update(1)  # 更新进度条
        try:
            edge = edgelist[index]  # 获取当前边
            index += 1
            e, dic = edge2weight(*edge)  # 将边的信息转换为权重
            if not tmpG.has_edge(e[0], e[1]):  # 如果tmpG中没有这条边，则跳过
                continue
            tmpG.remove_edge(e[0], e[1])  # 从tmpG中移除这条边
            G1.add_edge(*e, **dic)  # 将这条边添加到G1中
            C1 = []  # 初始化C1
            C1_list = list(nx.connected_components(G1))  # 获取G1的连通分量
            for i in C1_list:  # 找到包含q的连通分量
                if q in i:
                    C1 = i
            C1G = G1.subgraph(C1).copy()  # 创建C1的子图 C1是节点列表，C1G是对应节点列表的连通子图
            if C1G.number_of_edges() == pre_len or not lemma1(C1G) or not lemma2(C1G):  # 检查是否满足引理1和引理2
                continue
            if C1G.number_of_edges() >= 1.05*pre_len:  # 如果新的子图的边数比之前的多5%（QUESTION：pre_len的作用是什么，为什么要1.05倍更新一次？为什么如果满足了就跳出了？）
                pre_len = C1G.number_of_edges()  # 更新pre_len
            else:
                continue  # 如果不满足条件，则跳过
            C1G = get_alpha_beta_core_new_include_q(C1G, q, alpha, beta)  # 获取包含q的α-β核心，从连通子图中获取包含q的α-β核心
            if C1G.has_node(q):  # 如果C1G中包含q
                return peeling(C1G, q, alpha, beta, dim)  # 调用peeling函数
        except:
            break  # 如果出现异常，则跳出循环
    C1 = []  # 初始化C1
    C1_list = list(nx.connected_components(G1))  # 获取G1的连通分量
    for i in C1_list:  # 找到包含q的连通分量
        if q in i:
            C1 = i
    C1G = G1.subgraph(C1).copy()  # 创建C1的子图
    C1G = get_alpha_beta_core_new_include_q(C1G, q, alpha, beta)  # 获取包含q的α-β核心
    if C1G.has_node(q):  # 如果C1G中包含q
        return peeling(C1G, q, alpha, beta, dim)  # 调用peeling函数
    return None  # 如果没有找到满足条件的子图，则返回None

# 导入必要的库
import copy
import networkx as nx
from tqdm import tqdm

# 定义expand2D函数，接受图G、参数alpha、beta、q，以及可选的固定边集合fixed_edges
def expand2D(G, q, alpha, beta, dim=2, fixed_edges=None):
    # 深复制图G，以便在扩展过程中不改变原始图
    tmpG = copy.deepcopy(G)
    # 创建一个新的空图G1
    G1 = nx.Graph()
    # 从图G中获取与节点q相关联的边，并按权重降序排序
    edges_contain_q = [(u, v, data) for u, v, data in tmpG.edges(data=True) if q in (u, v)]
    node_edges = sorted(edges_contain_q, key=lambda x: x[2]['weight{}'.format(2)], reverse=True)
    # 根据q的值确定阈值limt
    limt = alpha if q.startswith('1') else beta
    # 如果排序后的边集合长度大于阈值limt，则找到第limt大的权重值
    # 否则，使用集合中最大的权重值（QUESTION：如果此处是else，那么不会永远满足不了αβ吗？）
    if len(node_edges) >= limt:
        th_largest_value = node_edges[limt-1][2]['weight{}'.format(2)]
    else:
        th_largest_value = node_edges[-1][2]['weight{}'.format(2)]
    # 打印找到的阈值
    print('find q th val', th_largest_value)
    
    # 对图中的所有边按权重降序排序
    edgelist = sorted(list(tmpG.edges(data=True)), key=lambda x: x[2]['weight{}'.format(2)], reverse=True)
    # 初始化索引idx
    idx = 0
    # 遍历排序后的边列表，找到第一个权重值小于等于阈值的边的索引
    while idx < len(edgelist) and edgelist[idx][2]['weight{}'.format(2)] > th_largest_value:
        idx += 1
        
    # 将权重值大于阈值的边添加到图G1中
    G1.add_edges_from(edgelist[:idx])
    # 创建一个新的空图S
    S = nx.Graph()
    # 初始化结果列表R
    R = []
    # 打印图中所有边的数量
    print('number of all edges:', len(edgelist))
    
    # stage 1 遍历边列表，在原来以q为中心生成的基础上加上其他的边，最终可以做到满足αβ的二部图
    # 创建一个进度条，总长度为边列表的长度，初始位置为idx
    pbar = tqdm(total=len(edgelist), initial=idx)
    # 遍历剩余的边列表
    while idx < len(edgelist) and S.number_of_edges() == 0:
        # 将当前边添加到图G1中
        G1.add_edges_from([edgelist[idx]])
        # 更新索引
        idx += 1
        # 更新进度条
        pbar.update(1)
        # 调用get_alpha_beta_core_new_include_q函数获取alpha和beta核心
        tmp = get_alpha_beta_core_new_include_q(G1, q, alpha, beta)
        # 如果核心图的边数为0，则继续
        if tmp.number_of_edges() == 0:
            continue
        # 如果提供了固定边集合，则检查核心图中是否包含这些边
        if fixed_edges is not None:
            for i in fixed_edges:
                if not tmp.has_edge(i):
                    break
        # 更新核心图S
        S = tmp
    # 关闭进度条
    pbar.close()
    # 如果核心图S的边数仍然为0，则返回空列表
    if S.number_of_edges() == 0:
        return []
    # 调用peeling函数获取剥离结果，先剥离1维，再剥离2维，拿到一个基准线
    firR = peeling(S, q, alpha, beta, 1, constrict=None, F=fixed_edges)
    # 调用gpeel1D2f函数获取1D剥离结果
    f1 = gpeel1D2f(firR, 1)
    # 将结果添加到列表R中
    R.append((f1, gpeel1D2f(firR, 2)))
    # 打印找到的第一个结果
    print('first R has been found.', R)
    
    # stage 2
    # 初始化限制字典constrict
    constrict = {'1':0, '2':0}
    # 设置限制条件，先对
    constrict['2'] = 0
    constrict['1'] = f1 + 0.01
    # 调用peeling函数获取2D剥离结果，找到了f2的上限，之后逐渐降低
    f2 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 2, constrict, fixed_edges), 2)
    
    # 循环直到f2为0
    while f2 > 0:
        # 更新限制条件
        constrict['1'] = 0
        constrict['2'] = f2
        # 调用peeling函数获取1D剥离结果
        f1 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 1, constrict, fixed_edges), 1)
        # 如果f1为-1，则跳出循环
        if f1 == -1:
            break
        # 检查结果是否已经存在于列表R中
        for i in R:
            if f1 <= i[0] and f2 <= i[1]:
                return R
        # 将新结果添加到列表R中
        R.append((f1, f2))
        # 记录当前凝聚子图
        logger.info('当前凝聚子图：{}'.format(R))
        # 更新限制条件
        constrict['2'] = 0
        constrict['1'] = f1 + 0.01
        # 调用peeling函数获取2D剥离结果
        f2 = gpeel1D2f(peeling(tmpG, q, alpha, beta, 2, constrict, fixed_edges), 2)
    # 返回结果列表R
    return R

def expand3D(G, q, alpha, beta, dim=3, F=[]):
    # 初始化结果列表
    R = []
    # 创建优先队列
    que = PriorityQueue()
    # 获取图G的所有边和权重
    edgelist = [i for i in G.edges(data=True)]
    # 初始化两个维度的天际线列表
    skyline1 = []
    skyline2 = []
    # 初始化最大值MAX
    MAX = 0
    # 尝试对图G进行3D扩展
    if expand(G, q, alpha, beta,  3) is None:
        return R
    # 如果3D扩展后的图不为空
    if len(expand(G, q, alpha, beta, 3).edges) != 0:
        # 计算3D扩展图的天际线值，并更新MAX
        maxf3 = community2skylineValue(expand(G, q, alpha, beta, 3), 3)[2]
        MAX = maxf3
        que.put((MAX-maxf3, (0, 0)))
    # 初始化记录点列表
    record_point = []
    # 初始化前一轮和当前轮的f3值
    pre_f3 = -1
    pre_round_f3 = -1
    # 当优先队列不为空时循环
    while not que.empty():
        S = []
        # 当优先队列不为空时循环
        while not que.empty():
            # 从队列中取出一个元素
            tmp_tup = que.get()
            # 计算当前的f3值
            f3 = MAX - tmp_tup[0]
            # 如果当前f3值与上一轮的f3值相同，则跳过
            if f3 == pre_round_f3:
                continue
            # 如果f3值改变，并且记录点列表为空，则添加当前点
            if f3 != pre_f3 and len(record_point) == 0:
                record_point.append(tmp_tup[1])
                pre_f3 = f3
            # 如果f3值未改变，则添加当前点
            elif f3 == pre_f3:
                record_point.append(tmp_tup[1])
            # 如果f3值改变，将当前点放回队列，并跳出循环
            else:
                que.put(tmp_tup)
                break
        # 更新本轮的f3值
        f3 = pre_f3
        pre_round_f3 = f3
        pre_f3 = -1
        # 去重复点
        record_point = list(set(record_point))
        # 遍历记录点
        for item in record_point:
            # 创建临时图
            tmpG = nx.Graph()
            # 获取记录点的两个维度值
            f1 = item[0]
            f2 = item[1]
            # 深拷贝F列表
            tmpF = copy.deepcopy(F)
            # 遍历所有边
            for edge in edgelist:
                # 如果边的权重满足条件，则添加到临时图中
                if edge[2]['weight{}'.format(3)] >= f3 and edge[2]['weight{}'.format(2)] > f2 and edge[2]['weight{}'.format(1)] > f1:
                    e, dic = edge2weight(*edge)
                    tmpG.add_edge(*e, **dic)
                # 如果边的权重等于f3，则添加到tmpF列表中
                if edge[2]['weight{}'.format(3)] == f3:
                    tmpF.append((edge[0], edge[1]))
            # 对临时图进行2D扩展
            S.extend(expand2D(tmpG, q, alpha, beta,  tmpF))
            # 遍历2D扩展的结果
            for tf1, tf2 in S:
                flag = 0
                # 检查是否在天际线内
                for f1s, f2s in zip(skyline1, skyline2):
                    if tf1 >= f1s and tf2 >= f2s:
                        if tf1 == f1s and tf2 == f2s:
                            flag = 1
                        else:
                            skyline1.remove(f1s)
                            skyline2.remove(f2s)
                # 如果不在天际线内，则添加到结果列表R中
                if flag == 0:
                    R.append((tf1, tf2, f3))
                    skyline1.append(tf1)
                    skyline2.append(tf2)
            # 更新天际线列表
            new_sky1 = skyline1[:]
            new_sky2 = skyline2[:]
            new_sky1.append(0)
            new_sky2.append(0)
            new_sky2.sort(reverse=True)
            new_sky1.sort()
            # 遍历新的天际线点
            for point in zip(new_sky1, new_sky2):
                # 创建新的临时图
                tmpG = nx.Graph()
                # 遍历所有边
                for edge in edgelist:
                    # 如果边的权重满足条件，则添加到临时图中
                    if edge[2]['weight{}'.format(1)] > point[0] and edge[2]['weight{}'.format(2)] > point[1]:
                        e, dic = edge2weight(*edge)
                        tmpG.add_edge(*e, **dic)
                # 计算临时图的天际线值
                skyline_list = community2skylineValue(expand(tmpG, q, alpha, beta, 3), 3)
                # 如果天际线列表长度大于等于3，则更新队列
                if len(skyline_list) >= 3:
                    maxdim3 = skyline_list[2]
                    que.put((MAX-maxdim3, (point[0], point[1])))
            # 打印当前结果
            print(R)
        # 清空记录点列表
        record_point = []
    # 去重复结果
    R = list(set(R))
    # 返回结果列表
    return R

def updatecpoint(C, s, d):
    # 初始化结果集合
    resC = []
    # 将输入的集合C转换为列表形式，以便于修改
    C = list(map(list, C))
    # 初始化Chat集合，用于存放未被支配的点
    Chat = []
    # 遍历集合C中的每个点c
    for c in C:
        # 初始化标志位，用于判断点c是否被支配
        flag = 1
        # 遍历点c和参考点s的每个维度
        for a, b in zip(c, s):
            # 如果点c在任一维度上小于参考点s，则该点被支配，设置标志位为0并跳出循环
            if a > b:
                flag = 0
                break
        # 如果点c没有被支配，则将其添加到Chat集合中
        if flag == 1:
            Chat.append(c)
    # 从集合C中移除Chat中的点，得到被支配的点集
    C = list(filter(lambda x: x not in Chat, C))
    # 将被支配的点集转换为元组形式
    C = list(map(tuple, C))
    # 对于给定的维度d，进行迭代
    for i in range(0, d):
        # 初始化临时结果集合
        restmpc = []
        # 初始化临时Chat集合
        ctmp = []
        # 深拷贝当前Chat集合
        Chat1 = copy.deepcopy(Chat)
        # 遍历Chat1中的每个点
        for c in Chat1:
            # 将当前点在第i维度的值更新为参考点s在第i维度的值
            c[i] = s[i]
            # 将更新后的点添加到临时集合ctmp中
            ctmp.append(c)
        # 将ctmp集合中的点转换为元组形式，并去重
        ctmp = list(set(map(tuple,ctmp)))
        # 遍历临时集合ctmp中的每个点
        for j in range(len(ctmp)): # 被支配
            # 初始化标志位，用于判断点ctmp[j]是否被支配
            isd = False
            # 遍历临时集合ctmp中的每个点
            for k in range(len(ctmp)):
                # 如果是同一个点，则跳过
                if j == k:
                    continue
                # 初始化标志位，用于判断点ctmp[j]是否支配点ctmp[k]
                dflag = True
                # 比较两个点的每个维度
                for x, y in zip(ctmp[j], ctmp[k]):
                    # 如果点ctmp[j]在任一维度上小于点ctmp[k]，则设置标志位为False并跳出循环
                    if x < y:
                        dflag = False
                        break
                # 如果点ctmp[j]支配点ctmp[k]，则设置isd标志位为True并跳出循环
                if dflag == True:
                    isd = True
                    break
            # 如果点ctmp[j]没有被支配，则将其添加到临时结果集合restmpc中
            if isd == False:
                restmpc.append(ctmp[j])
        # 将临时结果集合中的点添加到最终结果集合
            


def expandHighD(G, q, alpha, beta,  d, F=[]):
    # 如果维度d为3，则调用expand3D函数进行处理
    if d == 3:
        return expand3D(G, q, alpha, beta,  F)
    # 初始化结果列表
    R = []
    # 初始化全局天际线列表
    globalS = []
    # 初始化C集合，用于存放当前维度的点
    C = [tuple([0 for _ in range(d-1)])]
    # 创建优先队列
    que = PriorityQueue()
    # 获取图G的所有边和权重
    edgelist = [i for i in G.edges(data=True)]
    # 初始化最大值MAX
    MAX = 0
    # 如果d维扩展后的图不为空
    if len(expand(G, q, alpha, beta,  d).edges) != 0:
        # 计算d维扩展图的天际线值，并更新MAX
        maxfd = community2skylineValue(expand(G, q, alpha, beta,  d), d)[d-1]
        MAX = maxfd
        # 将初始点(0...0)放入优先队列
        tmptuple = tuple([0 for _ in range(d-1)])
        que.put((MAX-maxfd, tmptuple))
    # 初始化记录点列表
    record_point = []
    # 初始化前一轮和当前轮的fd值
    pre_fd = -1
    pre_round_fd = -1
    # 当优先队列不为空时循环
    while not que.empty():
        S = []
        # 当优先队列不为空时循环
        while not que.empty():
            # 从队列中取出一个元素
            tmp_tup = que.get()
            # 计算当前的fd值
            fd = MAX - tmp_tup[0]
            # 如果当前fd值与上一轮的fd值相同，则跳过
            if fd == pre_round_fd:
                continue
            # 如果fd值改变，并且记录点列表为空，则添加当前点
            if fd != pre_fd and len(record_point) == 0:
                record_point.append(tmp_tup[1])
                pre_fd = fd
            # 如果fd值未改变，则添加当前点
            elif fd == pre_fd:
                record_point.append(tmp_tup[1])
            # 如果fd值改变，将当前点放回队列，并跳出循环
            else:
                que.put(tmp_tup)
                break
        # 更新本轮的fd值
        fd = pre_fd
        pre_round_fd = fd
        pre_fd = -1
        # 去重复点
        record_point = list(set(record_point))
        # 遍历记录点
        for item in record_point:
            # 创建临时图
            tmpG = nx.Graph()
            # 深拷贝F列表
            tmpF = copy.deepcopy(F)
            # 遍历所有边
            for edge in edgelist:
                # 获取边的前d-1维权重
                attri = [edge[2]['weight{}'.format(i+1)] for i in range(d-1)]
                # 检查边的权重是否满足条件
                attriflag = True
                for ea, limit in zip(attri, item):
                    if ea <= limit:
                        attriflag = False
                        break
                # 如果边的d维权重满足条件，并且前d-1维权重大于等于item对应值，则添加到临时图中
                if edge[2]['weight{}'.format(d)] >= fd and attriflag == True:
                    e, dic = edge2weight(*edge)
                    tmpG.add_edge(*e, **dic)
                # 如果边的d维权重等于fd，则添加到tmpF列表中
                if edge[2]['weight{}'.format(d)] == fd:
                    tmpF.append((edge[0], edge[1]))
            # 对临时图进行高维扩展
            S.extend(expandHighD(tmpG, q, alpha, beta,  d-1, tmpF))
            # 遍历高维扩展的结果
            for p in S:
                # 初始化标志位，用于判断点p是否被支配
                flag = 0
                # 遍历全局天际线列表globalS
                for gp in globalS:
                    # 检查点p是否支配点gp
                    checkbing = True
                    for i in range(len(p)):
                        if p[i] < gp[i]:
                            checkbing = False
                            break
                    # 如果点p支配点gp，或者与gp相等，则设置标志位
                    if checkbing == True:
                        if p == gp:
                            flag = 1
                        else:
                            globalS.remove(gp)
                # 如果点p没有被支配，则添加到结果列表R中，并更新全局天际线列表
                if flag == 0:
                    R.append(p+(fd,))
                    globalS.append(p)

                # 更新C集合
                C = updatecpoint(C, p, d-1)
            
            # 遍历C集合中的点
            for point in C:
                # 创建新的临时图
                tmpG = nx.Graph()
                # 遍历所有边
                for edge in edgelist:
                    # 获取边的前d-1维权重
                    attri = [edge[2]['weight{}'.format(i+1)] for i in range(d-1)]
                    # 检查边的权重是否满足条件
                    attriflag = True
                    for ea, limit in zip(attri, point):
                        if ea <= limit:
                            attriflag = False
                            break
                    # 如果边的前d-1维权重大于等于point对应值，则添加到临时图中
                    if attriflag == True:
                        e, dic = edge2weight(*edge)
                        tmpG.add_edge(*e, **dic)
                # 计算临时图的天际线值
                skyline_list = community2skylineValue(expand(tmpG, q, alpha, beta,  d), d)
                # 如果天际线列表长度大于等于d，则更新队列
                if len(skyline_list) >= d:
                    maxdimd = skyline_list[d-1]
                    que.put((MAX-maxdimd, point))
            # 打印当前结果
            print(R)
        # 清空记录点列表
        record_point = []
    # 去重复结果
    R = list(set(R))
    # 返回结果列表
    return R
        

if __name__ == "__main__":
    path = '/home/yaodi/luoxuanpu/crime4dim.txt'
    q = '1128'
    alpha, beta = 2, 2
    # path = './dim4graph.txt'
    # q = '22'
    # alpha, beta = 2, 3
    data = get_data(path)
    G = build_bipartite_graph(data)
    connect_subgraph = get_alpha_beta_core_new_include_q(G, q, alpha, beta)
    origG = G.subgraph(connect_subgraph).copy()
    nx.set_edge_attributes(G, 0, "visited")
    starttime = time.time()
    # print(GetCandVal(origG, q, alpha, beta,  3))
    # print(get_alpha_beta_core_new_include_q(G,q,alpha,beta))
    # pdb.set_trace()
    # print(peeling(origG, q, alpha, beta, 1))
    # print(expand(origG, q, alpha, beta,  1))
    print(expand2D(origG, alpha, beta, q))
    # print(peeling2D(origG, q, alpha, beta))
    # print(res)
    # logger.info(res)
    endtime = time.time()
    print(endtime - starttime)
    logger.info(endtime - starttime)