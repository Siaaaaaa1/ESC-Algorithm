import sqlite3

# 连接到 SQLite 数据库
# 数据库文件是 my_database.db
# 如果文件不存在，会自动在当前目录创建:

def generate_new_database(name, extra=False, N=None, path=None):
    conn = sqlite3.connect(name)
    cursor = conn.cursor()
    data = []

    if extra == False:
        with open(path, 'r') as f:
            for line in f.readlines():
                info = line.strip('\n').split(' ')
                data.append(info)
                
            l = len(data[0])
            tl = []
            for i in range(l-2):
                t = 'val{} REAL'.format(i+1)
                tl.append(t)
            create = ', '.join(tl)
    else:
        for line in N:
            info = line.strip('\n').split(' ')
            data.append(info)
            
        l = len(data[0])
        tl = []
        for i in range(l-2):
            t = 'val{} REAL'.format(i+1)
            tl.append(t)
        create = ', '.join(tl)
    
    # 定义删除表的 SQL 语句
    drop_table_sql = 'DROP TABLE IF EXISTS show_graph'

    # 执行 SQL 语句
    cursor.execute(drop_table_sql)

    # 提交事务
    conn.commit()
    cre = 'CREATE TABLE IF NOT EXISTS show_graph (src INTEGER, tar INTEGER, {})'.format(create)           
    # print(cre)     
    # 创建表
    cursor.execute(cre)

    # 插入数据
    for d in data:
        tmp = []
        for i in range(len(d)-2):
            tmp.append('val{}'.format(i+1))
        d = list(map(float, d))
        d[0] = int(d[0])
        d[1] = int(d[1])
        d = list(map(str, d))
        v = '('+', '.join(d)+')'
        # print("INSERT OR IGNORE INTO users {} VALUES {}".format('(src, tar, {})'.format(', '.join(tmp)), v))
        cursor.execute("INSERT OR IGNORE INTO show_graph {} VALUES {}".format('(src, tar, {})'.format(', '.join(tmp)), v))

    # 查询数据
    cursor.execute("SELECT * FROM show_graph")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    generate_new_database('path/path', 'my_database.db')