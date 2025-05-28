import sqlite3
import streamlit as st
import time
from paper_debug import build_bipartite_graph, get_alpha_beta_core_new_include_q, peeling, peeling2D, peeling3D, peelingHighD, gpeel1D2f
from paper_debug_expand import expand,expand2D,expand3D,expandHighD
from yanjiudian3 import generate_new_database
import matplotlib.pyplot as plt
import networkx as nx
import csv
import utils

# 连接到 SQLite 数据库
# 数据库文件是 my_database.db
# 如果文件不存在，会自动在当前目录创建:
# streamlit run /data/gaohaowen/workspace/1003/jiemian.py
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

def gpig(G, path):
    top_nodes, bottom_nodes = nx.bipartite.sets(G)
    edges = list(G.edges())
    # 生成布局
    pos = nx.bipartite_layout(G, nodes=top_nodes)
    plt.clf()
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_color='c', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=bottom_nodes, node_color='m', node_size=500)

    # 绘制边
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)

    # 添加标签
    nx.draw_networkx_labels(G, pos)

    plt.axis('off')

    # 保存图片
    plt.savefig(path)


def sky2G(G, res):
    edges = []
    for e in G.edges(data=True):
        flag = True
        for es, i in enumerate(res):
            if e[2]['weight{}'.format(es+1)]<i:
                flag = False
                break
        if flag == True:
            edges.append(e)
    return get_alpha_beta_core_new_include_q(G, q, alpha, beta)


st.markdown('### 多维边权二部图凝聚子图挖掘原型系统', unsafe_allow_html=True)

# uploaded_file = st.file_uploader("上传文件", type=["txt"])

# 创建一个下拉选项框

option = st.selectbox(
    '请选择原图',
    ('show', 'DBpedia','BookCrossing','Writers','Github','arXiv','Crime')
)


# if uploaded_file is not None:
#     # 获取上传文件的文件名
#     file_name = uploaded_file.name
#     st.write("上传的文件名:", file_name)

#     # 处理上传的文件
#     data = uploaded_file.readlines()
#     generate_new_database(file_name+'.db', extra=True, N=data)
#     options = ['show', file_name]

#     # 重新渲染下拉框
#     option = st.selectbox('请选择原图', options)
file_path = 'path/path'.format(option)
with open(file_path, 'r') as file:
    # 逐行读取
    data = file.readlines()
    generate_new_database("my_database.db", extra=True, N=data)
    options = ['show', option]
    # 重新渲染下拉框
    option = st.selectbox('请选择原图', options)

# 查询数据
cursor.execute("SELECT * FROM {}_graph".format(option))
rows = cursor.fetchall()
conn.commit()
conn.close()

def trans2G(rows):
    data = []
    for i in rows:
        upper = '1'+str(i[0])
        lower = '2'+str(i[1])
        l = list(i)[2:]
        dic = {}
        for index, v in enumerate(l):
            dic.update({'weight{}'.format(index+1):float(v)})
        data.append((upper, lower, dic))
    return build_bipartite_graph(data)

G = trans2G(rows)

# 输入框
dim = st.number_input('请输入属性值维度数', min_value=0, value=1)
alpha = st.number_input('请输入alpha值', min_value=0, value=1)
beta = st.number_input('请输入beta值', min_value=0, value=1)
q = st.number_input('请输入查询顶点编号', min_value=0, value=11)
orig_path = '/home/rabbit/workspace/1002/orig.png'
origG = G.copy()
if q > 10:
    q = str(q)
    connect_subgraph = get_alpha_beta_core_new_include_q(G, q, alpha, beta)
    origG = G.subgraph(connect_subgraph).copy()
    gpig(G, orig_path)

# 显示结果、耗时和图片的地方
result = st.empty()
time_taken = st.empty()
len_taken = st.empty()
image = st.empty()
col1, col2 = st.columns(2)
# 两个选择的按钮

res = st.radio(
    "请选择使用的算法：",
    ('剥离算法', '扩展算法')
)

# st.radio('hhh',[1,4])

res_path = '/home/rabbit/workspace/1002/res.png'
# 按钮
if st.button('计算'):
    if res == '剥离算法':
        if dim == 1:    
            start_time = time.time()
            text = gpeel1D2f(peeling(origG, q, alpha, beta, 1),1)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            ttxt = [(text, )]
            list_str = ', '.join([str(x) for x in ttxt])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)

            gpig(sky2G(origG, (text,)), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(ttxt)} 个')
        if dim == 2:    
            start_time = time.time()
            print(q,alpha,beta)
            text = peeling2D(origG, q, alpha, beta)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            
            list_str = ', '.join([str(x) for x in text])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)
            gpig(sky2G(origG, text[0]), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(text)} 个')
        if dim == 3:    
            start_time = time.time()
            text = peeling3D(origG, q, alpha, beta)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            list_str = ', '.join([str(x) for x in text])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)
            gpig(sky2G(origG, text[0]), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(text)} 个')
        if dim > 3:
            start_time = time.time()
            text = peelingHighD(origG, q, alpha, beta, dim)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            list_str = ', '.join([str(x) for x in text])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)
            gpig(sky2G(origG, text[0]), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(text)} 个')
    if res == '扩展算法':
        if dim == 1:    
            start_time = time.time()
            text = gpeel1D2f(expand(origG, q, alpha, beta, 1),1)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            ttxt = [(text, )]
            list_str = ', '.join([str(x) for x in ttxt])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)

            gpig(sky2G(origG, (text,)), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(ttxt)} 个')
        if dim == 2:
            start_time = time.time()
            text = expand2D(origG, q, alpha, beta, 2)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            list_str = ', '.join([str(x) for x in text])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)
            gpig(sky2G(origG, text[0]), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(text)} 个')
        if dim == 3:
            start_time = time.time()
            text = expand3D(origG, q, alpha, beta, 3)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            list_str = ', '.join([str(x) for x in text])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)
            gpig(sky2G(origG, text[0]), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(text)} 个')
        if dim > 3:
            start_time = time.time()
            text = expandHighD(origG, q, alpha, beta, dim)
            end_time = time.time()
            # 这里可以添加您的计算代码
            # 这里只是一个简单的示例
            list_str = ', '.join([str(x) for x in text])
            result_text = '结果: {}'.format(list_str)
            
            # 显示结果
            result.text(result_text)
            gpig(sky2G(origG, text[0]), res_path)
            time_taken.text(f'耗时: {end_time - start_time} 秒')
            len_taken.text(f'凝聚子图个数: {len(text)} 个')
    # 显示图片
    with col1:
        st.image(orig_path, caption='原始图')

    # 在第二列显示第二张图片
    with col2:
        st.image(res_path, caption='凝聚子图')