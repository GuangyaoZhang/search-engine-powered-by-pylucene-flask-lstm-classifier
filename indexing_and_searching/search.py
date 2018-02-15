import sys
sys.path.append("..")
from data_processing.data_processing import Text_processing
from classifier_.main import Classifier
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
import re
import os
class Searcher():
    def __init__(self):
        self.vm = lucene.initVM()
        self.text = Text_processing()
        self.text_used_for_taggin = Text_processing(just_segging = False)

        self.data_dir = self.text.get_data_dir()
        self.index_dir = os.path.join(self.data_dir,'index')

        self.analyzer = WhitespaceAnalyzer()
        self.keys = self.text.get_keys()

        self.directory = SimpleFSDirectory(Paths.get(self.index_dir))
        self.searcher = IndexSearcher(DirectoryReader.open(self.directory))

        self.classifier = Classifier()

        self.reT = re.compile(r"((Title|PubDate|WBSB|DSRXX|SSJL|AJJBQK|CPYZ|PJJG|WBWB)( *)(:|：)(.*?)(?=$|(OR|AND| )( +)(Title|PubDate|WBSB|DSRXX|SSJL|AJJBQK|CPYZ|PJJG|WBWB)))",re.I)

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


    def return_in_mysql(fun):
        def wrapper(self, command, num, use_clf):
            # if(use_clf):
            results, probs, key_use = fun(self,command,num,use_clf)
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
            # else:
            #     results = fun(self,command,num,use_clf)
            #     ds = []
            #     for i in results:
            #         d = dict()
            #         id = int(i['id'])
            #         raw_data = self.text.select_from_mysql(id)
            #         for key, content in zip(self.keys, raw_data):
            #             d[key] = content
            #         ds.append(d)
            #     return ds
        return wrapper

    @return_in_mysql
    def search(self,command,num,use_clf):
        print("log1",command,num,use_clf)
        self.vm.attachCurrentThread()
        searcher = self.searcher

        print("command",command)

        if (not self.reT.search(command)):
            if (use_clf):
                print("sentence feed to classify",command)
                probs = self.classifier.classify(command)
                command = self.text.seg(command)
                command = self.text.remove_stop_word(command)
                # command = self.text.replace_white_space_with_dash(command)
                key = sorted(range(len(self.keys)),key=lambda i:probs[i],reverse=True)
                key_use = []
                key_use.append(key[0])
                for i in key[1:]:
                    if probs[i]>0.3 or probs[i]-probs[key[0]]>-0.1:
                        key_use.append(i)

                command_final = self.keys[key_use[0]]+":("+command+")"
                for i in key_use[1:]:
                    command_final = "%s OR %s:(%s)"% (command_final,self.keys[i],command)
                command=command_final

                # command = "Title:\"2016 吉 07 民终 491号 包颜峰诉\""
                # command = "PubDate:\"2016 11 24\""
                # command = "WBSB:浙江省 WBSB:苍南县 WBSB:人民法院"
                print(command)
                # command = "Title:陕西省-高级-人民法院 Pubdate:陕西省-高级-人民法院"
                query = QueryParser("PubDate",WhitespaceAnalyzer()).parse(command)
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
                return results,[None]*len(self.keys),self.keys
        else:
            print('command',command)
            ps = self.reT.findall(command)
            print(ps)
            print(type(command))
            rem= self.reT.sub(command,' ')
            print(ps)
            print(rem)
            q_t = []
            key_use = []
            for i in ps:

                f = i[1]
                data = i[4]
                rela = i[5]

                key_use.append(f)

                q_t.append(f)
                q_t.append(':')
                seg_t = self.text.seg(data)
                seg_t = self.text.remove_stop_word(seg_t)
                dash_t = self.text.replace_white_space_with_dash(seg_t)
                q_t.append(dash_t)
                if(rela):
                    q_t.append(" %s "%rela)
                print('tract pattern',q_t)
            q_f = "".join(q_t)
            print("final q",q_f)
            query = QueryParser("PubDate", SimpleAnalyzer()).parse(q_f)
            print("query",query)
            scoreDocs = searcher.search(query, num).scoreDocs

            results = []

            for scoreDoc in scoreDocs:
                doc = searcher.doc(scoreDoc.doc)
                result = dict()
                for i in self.keys:
                    result[i] = doc.get(i)
                result['id'] = doc.get('id')
                results.append(result)
            return results,[None]*len(key_use),key_use

    def query(self,command):
        self.vm.attachCurrentThread()
        command_raw = command
        command = self.text_used_for_taggin.seg(command)
        print(self.text._lib)
        print(self.text_used_for_taggin._lib)

        Key_words = ['when','where','what']
        Stop_words = ['发生','有']
        Stop_words_set  = set(Stop_words)
        when = r'\w+(?=_t)'
        where = r'\w+(?=_ns)'
        what = r'\w+(?=_v(?= |$))|\w+(?=_n(?= |$))|\w+(?=_a(?= |$))'
        results = r'判多少年|怎么判|判处\w+吗'
        counts = r'有多少|发生过多少|有过\w+吗'


        pattern = dict()
        print(command)
        pattern['when'] = re.findall(when,command)
        pattern['where'] = re.findall(where,command)
        pattern['what'] = re.findall(what,command)
        pattern['what'] = [i for i in pattern['what'] if i not in Stop_words_set]

        print('command',command)

        if(re.search(results,command_raw)):
            pattern['qustion'] = 'results'
            print("results")
        if(re.search(counts,command_raw)):
            pattern['qustion'] = 'counts'
            print("count")
        print(pattern)
        return pattern


if __name__ == "__main__":
    sear = Searcher()
    # sear.indexing()
    t = Text_processing()
    print()
    a=sear.search('你好啊:中国 Title： 杀人  OR    PubDate: 我 ous:kdka ',1,False)

    for i in a:
        print(i)
        id = int(i['id'])
        print(t.select_from_mysql(id))



