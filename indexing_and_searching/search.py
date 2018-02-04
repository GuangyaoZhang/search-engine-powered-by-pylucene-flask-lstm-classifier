import sys
sys.path.append("..")
from data_processing.data_processing import Text_processing
from classifier.classifier import Classifier
import lucene
from org.apache.lucene.queryparser.classic import QueryParser
from java.nio.file import Paths
from java.util import Collection
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

from org.apache.lucene.analysis.cjk import CJKAnalyzer
from org.apache.lucene.util import Version

from org.apache.lucene.analysis import CharArraySet
from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer

from org.apache.lucene.store import SimpleFSDirectory
from java.nio.file import Paths

from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.analysis.cjk import CJKAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.standard import StandardAnalyzer

from org.apache.lucene.analysis import CharArraySet

from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer

from org.apache.lucene.queryparser.classic import MultiFieldQueryParser

from org.apache.lucene.queryparser.classic import QueryParserBase

from org.apache.lucene.search import BooleanClause


from org.apache.lucene.analysis.core import SimpleAnalyzer

import os
class Searcher():
    def __init__(self):
        self.vm = lucene.initVM()
        self.text = Text_processing()

        self.data_dir = self.text.get_data_dir()
        self.index_dir = os.path.join(self.data_dir,'index')

        self.analyzer = WhitespaceAnalyzer()
        self.keys = self.text.get_keys()

        self.directory = SimpleFSDirectory(Paths.get(self.index_dir))
        self.searcher = IndexSearcher(DirectoryReader.open(self.directory))

        self.classifier = Classifier()


    def indexing(self):
        docs = self.text.load_seg_without_stopword_data()

        if(os.path.exists(self.index_dir)):
            r = input("Indexing Dir has existed! Continue indexing?")
            if(r.lower()!='y'):
                return -1

        if (not os.path.exists(self.index_dir)):
            os.makedirs(self.index_dir)
        store = SimpleFSDirectory(Paths.get(self.index_dir))

        # todo
        # version.LUCENE_6_5_0
        # analyzer = CJKAnalyzer(CharArraySet.EMPTY_SET)
        # analyzer =SmartChineseAnalyzer()

        # analyzer = StandardAnalyzer(Version.LUCENE_6_5_0)
        # index_writer = IndexWriter(store,analyzer,True,IndexWriter.MaxFieldLength(512))
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        id_conf = FieldType()
        id_conf.setStored(True)
        id_conf.setTokenized(False)
        id_conf.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        date_conf = FieldType()
        date_conf.setStored(True)
        date_conf.setTokenized(True)
        date_conf.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        content_conf = FieldType()
        content_conf.setStored(True)
        content_conf.setTokenized(True)
        content_conf.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        for n, i in enumerate(docs):
            document = Document()

            for key, content in i.items():
                if (key == 'PubDate'):
                    document.add(Field(key, content, date_conf))
                else:
                    document.add(Field(key, content, content_conf))
            document.add(Field('id',str(n),id_conf))
            writer.addDocument(document)
            if (n % 1000 == 0):
                print(n)

        writer.commit()
        writer.close()
    def search_data(self,command,num,use_clf):
        print("log1",command,num,use_clf)
        self.vm.attachCurrentThread()
        searcher = self.searcher

        s = set(command)

        if(":" not in s and "：" not in s):
            if (use_clf):
                probs = self.classifier.predict(command)
                command = self.text.seg(command)
                command = self.text.remove_stop_word(command)
                command = self.text.replace_white_space_with_dash(command)
                key = sorted(range(len(self.keys)),key=lambda i:probs[i],reverse=True)
                key_use = []
                key_use.append(key[0])
                for i in key[1:]:
                    if probs[i]>0.3 or probs[i]-probs[key[0]]>-0.1:
                        key_use.append(i)

                # command_final = self.keys[key_use[0]]+":\""+command+"\""
                # # command_final = "Title" + ":" + command
                # for i in key_use[1:]:
                #     command_final = "%s OR %s:%s"% (command_final,self.keys[i],command)
                # command=command_final

                # command = "Title:\"2016 吉 07 民终 491号 包颜峰诉\""
                # command = "PubDate:\"2016 11 24\""
                # command = "WBSB:浙江省 WBSB:苍南县 WBSB:人民法院"
                print("矣")
                print(command)
                command = "Title:陕西省-高级-人民法院 Pubdate:陕西省-高级-人民法院"
                query = QueryParser("PubDate",SimpleAnalyzer()).parse(command)
                # parser =  MultiFieldQueryParser(['WBSB'], self.analyzer)
                # parser.setDefaultOperator(QueryParserBase.AND_OPERATOR)
                # query =parser.parse(QueryParserBase,command)



                # P = QueryParser('Pubdate', CJKAnalyzer())
                # query = MultiFieldQueryParser(['WBSB','Pubdate'],CJKAnalyzer()).parse(P,command)
                #
                #
                # # query = MultiFieldQueryParser(['WBSB',"title"], CJKAnalyzer()).getMultiFieldQuery(q)
                # # p = QueryParser('Title', CJKAnalyzer()).parse("你好 中国 你好 北京")
                # print(query)

                # fields = []
                # # fields = ["filename", "contents", "description"]
                #
                # for i in key_use:
                #     fields.append(self.keys[i])
                # flags = [BooleanClause.Occur.SHOULD]*len(fields)
                #
                # query=MultiFieldQueryParser.parse(command, fields, flags, WhitespaceAnalyzer())
                #
                print(query)

                scoreDocs = searcher.search(query, num).scoreDocs

                results = []

                for scoreDoc in scoreDocs:
                    doc = searcher.doc(scoreDoc.doc)
                    result = dict()
                    for i in self.keys:
                        result[i] = doc.get(i)
                    result['id'] = doc.get('id')
                    results.append(result)
                probs_tmp = ""
                for key,prob in zip(self.keys,probs):
                    probs_tmp+="%s:%2f "%(key,prob)
                probs = probs_tmp
                key_use_tmp = ""
                for i in key_use:
                    key_use_tmp+="%s "%(self.keys[i])
                key_use=key_use_tmp
                return results,probs,key_use

            else:
                command = self.text.seg(command)
                command = self.text.remove_stop_word(command)
                fields = self.keys
                flags = [BooleanClause.Occur.SHOULD] * len(fields)

                query = MultiFieldQueryParser.parse(command, fields, flags, WhitespaceAnalyzer())

                # command_final = "Title:"+command
                # for i in self.keys[1:]:
                #     command_final = "%s OR %s:%s"% (command_final,i,command)
                # command=command_final
                # print("矣")
                # print(command)
                # query = QueryParser("Title", self.analyzer).parse(command)

                fields = self.keys
                flags = [BooleanClause.Occur.SHOULD] * len(fields)

                query = MultiFieldQueryParser.parse(command, fields, flags, WhitespaceAnalyzer())
                print(query)
                scoreDocs = searcher.search(query, num).scoreDocs

                results = []

                for scoreDoc in scoreDocs:
                    doc = searcher.doc(scoreDoc.doc)
                    result = dict()
                    for i in self.keys:
                        result[i]=doc.get(i)
                    result['id'] = doc.get('id')
                    results.append(result)
                return results

    def search(self,command,num,use_clf):
        if(use_clf):
            results, probs, key_use = self.search_data(command,num,use_clf)
            print("result",results)
            ds = []
            for i in results:
                d = dict()
                id = int(i['id'])
                raw_data = list(self.text.select_from_mysql(id)[0][1:])
                raw_data[1] = raw_data[1].strftime('%y-%m-%d')
                print(raw_data)
                for key,content in zip(self.keys,raw_data):
                    d[key] = content
                ds.append(d)
                print(d)
            return ds,probs, key_use
        else:
            results = self.search_data(command, num, use_clf)
            ds = []
            for i in results:
                d = dict()
                id = int(i['id'])
                raw_data = self.text.select_from_mysql(id)
                for key, content in zip(self.keys, raw_data):
                    d[key] = content
                ds.append(d)
            return ds


if __name__ == "__main__":
    sear = Searcher()
    # sear.indexing()
    t = Text_processing()
    print()
    a=sear.search("陕西省高级人民法院",1,True)[0]

    for i in a:
        print(i)
        id = int(i['id'])
        print(t.select_from_mysql(id))



