# from data_processing import Text_processing
#
# t = Text_processing(60)
# a = t.seg("2015 13 31")
# print(a)
# -*- coding: UTF-8 -*
import pymysql



from data_processing import Text_processing

import re
import sys
def insert():
    db = pymysql.connect("localhost", "root", "19961231", "law_data", charset='utf8')
    keys = ['Title', 'PubDate', 'WBSB', 'DSRXX', 'SSJL', 'AJJBQK', 'CPYZ', 'PJJG', 'WBWB']
    cursor = db.cursor()
    sql = """drop table law_data;
    CREATE TABLE law_data(
    id int (10) PRIMARY KEY ,
    Title LONGTEXT,
    Pubdate DATE ,
    WBSB LONGTEXT,
    DSRXX LONGTEXT,
    SSJL LONGTEXT,
    AJJBQK LONGTEXT,
    CPYZ LONGTEXT,
    PJJG LONGTEXT,
    WBWB LONGTEXT) default charset=utf8"""
    cursor.execute(sql)
    pattern = re.compile(r'\d\d\d\d-\d\d-\d\d')
    text = Text_processing()
    raw_data = text.load_doc_data()
    sql_base = """insert into law_data(id,%s,%s,%s,%s,%s,%s,%s,%s,%s)""" % tuple(keys)
    for num,i in enumerate(raw_data):
        value = []
        value.append(num)
        for key in keys:
            if(key in i):
                if(key=="PubDate"):
                    if(not pattern.match(i[key])):
                        value.append('NULL')
                        continue
                value.append("\""+i[key]+"\"")
            else:
                value.append('NULL')

        sql=sql_base+" value(%d,%s,%s,%s,%s,%s,%s,%s,%s,%s)"%tuple(value)

        cursor.execute(sql)
        if(num%100==0):
            print(num)
    db.commit()
    cursor.close()
    db.close()
def select():
    db = pymysql.connect("localhost", "root", "19961231", "law_data", charset='utf8')
    keys = ['Title', 'PubDate', 'WBSB', 'DSRXX', 'SSJL', 'AJJBQK', 'CPYZ', 'PJJG', 'WBWB']
    cursor = db.cursor()
    sql = """ select * from law_data where id=1"""
    cursor.execute(sql)
    print(cursor.fetchall())
    db.close()
# insert()
select()


