from gensim.models import Word2Vec
import data_processing
#
file_in = open('/media/zgy/新加卷/cuda/zhwiki/zhwiki_2017_03.clean','r')
file_out = open('/media/zgy/新加卷/cuda/zhwiki/zhwiki_2017_03.seg1','w')
file_in = file_in.readlines()
lens = len(file_in)

import time
a = time.time()
t = data_processing.Text_processing()
for num,sentence in enumerate(file_in):
    seg_sentence = t.seg(sentence)
    file_out.write(seg_sentence)
    file_out.write("\n")
    if(num%100==0):
        print(num/lens*100)

b = time.time()
print(b-a)
file_out.close()
#
# import logging
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# from gensim.models import Word2Vec
#
# file_in = open('/media/zgy/新加卷1/cuda/zhwiki/zhwiki_2017_03.seg','r')
#
# file_in = file_in.readlines()
#
# print("start train")
#
# class it():
#     def __init__(self):
#         pass
#     def __iter__(self):
#         for i in file_in:
#             yield i.split()
#
# i = it()
# model = Word2Vec(i,size=200,min_count=10,workers=8)
# model.save('/media/zgy/新加卷1/cuda/zhwiki/model')
#
# model = Word2Vec.load('/media/zgy/新加卷1/cuda/zhwiki/model')

model = Word2Vec.load('/media/zgy/新加卷1/word_vec/Word60.model' )
print(len(model["中华人民共和国"]))
