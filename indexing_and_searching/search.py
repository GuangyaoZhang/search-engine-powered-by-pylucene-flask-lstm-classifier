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
# from org.apache.lucene.analysis import WhitespaceAnalyzer

from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer


import os
class Searcher():
    def __init__(self):
        self.vm = lucene.initVM()
        self.text = Text_processing()

        current_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(current_dir, '../../data')
        self.index_dir = os.path.join(self.data_dir,'index')
        self.analyzer = WhitespaceAnalyzer()
        self.keys = ['Title', 'PubDate', 'WBSB', 'DSRXX', 'SSJL', 'AJJBQK', 'CPYZ', 'PJJG', 'WBWB']

        self.classifier = Classifier()


    def indexing(self):
        docs = self.text.load_seg_without_stopword_data()

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

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        for n, i in enumerate(docs):
            document = Document()

            for key, content in i.items():
                if (key == 'PubDate'):
                    document.add(Field(key, content, t1))
                else:
                    document.add(Field(key, content, t2))
            document.add(Field('id',str(n),t1))
            writer.addDocument(document)
            if (n % 1000 == 0):
                print(n)

        writer.commit()
        writer.close()
    def search(self,command,num,use_clf):
        self.vm.attachCurrentThread()
        directory = SimpleFSDirectory(Paths.get(self.index_dir))
        searcher = IndexSearcher(DirectoryReader.open(directory))

        s = set(command)

        if(":" not in s and "：" not in s):
            if (use_clf):

                probs = self.classifier.predict(command)
                key = sorted(range(len(self.keys)),key=lambda i:probs[i],reverse=True)
                key_use = []
                key_use.append(key[0])
                for i in key[1:]:
                    if probs[i]>0.3 or probs[i]-probs[key[0]]>-0.1:
                        key_use.append(i)

                command_final = self.keys[key_use[0]]+":"+command
                for i in key_use[1:]:
                    command_final = "%s OR %s:%s"% (command_final,self.keys[i],command)
                command=command_final

                print("矣")
                print(command)
                query = QueryParser("Title", self.analyzer).parse(command)
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
                for i in key_use[::-1]:
                    key_use_tmp+="%s "%(self.keys[i])
                key_use=key_use_tmp
                return results,probs,key_use

            else:
                command_final = "Title:"+command
                for i in self.keys[1:]:
                    command_final = "%s OR %s:%s"% (command_final,i,command)
                command=command_final
                print("矣")
                print(command)
                query = QueryParser("Title", self.analyzer).parse(command)
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

if __name__ == "__main__":
    sear = Searcher()
    # sear.indexing()
    docs = sear.text.load_doc_data()
    print()
    a=sear.search("杀人",1,False)[0]
    print(a)
    print(docs[int(a['id'])])

