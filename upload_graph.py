import sqlite3
import utils
from yanjiudian3 import generate_new_database
# 将txt文件的数据集上传到sqlite上
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()
option = "graph"
file_path = 'path/path'.format(option)
with open(file_path, 'r') as file:
    # 逐行读取
    data = file.readlines()
    generate_new_database(option+'.db', extra=True, N=data)
