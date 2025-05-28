import sqlite3
import csv 

def new_table(txt_file, table_name, cursor, conn):
    with open(txt_file, 'r') as f:
        data = f.readlines()
    data = [i.strip().split() for i in data]
    table_name = 'graph'
    cursor.execute('''CREATE TABLE IF NOT EXISTS {} (
                   BEGIN INTEGER,
                   END INTEGER,
                   weight1 REAL,
                   weight2 REAL,
                   );'''.format(table_name))
    
    
    conn.commit()
    with open('users.txt', 'r') as file:
        # 读取文件内容
        lines = file.readlines()
    # 准备 SQL 插入语句
    insert_sql = 'INSERT INTO users (BEGIN, END, weight1, weight2) VALUES (?, ?, ?, ?)'
    # 遍历每一行数据
    for line in lines:
        # 去除行尾的换行符，并以逗号分隔字段
        begin, end, weight1, weight2 = line.strip().split(',')
        # 执行插入操作
        cursor.execute(insert_sql, (begin, end, weight1, weight2))
        

def csv_to_txt_no_header(csv_file_path, txt_file_path):
    """
    Convert a CSV file to a TXT file without headers.
    
    Parameters:
    csv_file_path (str): The path to the input CSV file.
    txt_file_path (str): The path to the output TXT file.
    """
    with open(csv_file_path, 'r') as csv_file, open(txt_file_path, 'w') as txt_file:
        csv_reader = csv.reader(csv_file)
        # Skip the header row
        next(csv_reader, None)
        # Write each row to the TXT file without headers, separated by spaces
        for row in csv_reader:
            txt_file.write(' '.join(row) + '\n')
