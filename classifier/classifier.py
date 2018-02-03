
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
# d = os.getcwd()
# os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))
from torch.utils.data import DataLoader
from models import TextLSTM
from models import TextCNN
from torch.autograd import Variable


from data_processing.data_processing import Text_processing
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix
import os

import jieba
import numpy as np
import gc
class clf_profile():
    def __init__(self):
        self.keys = ['Title','PubDate','WBSB', 'DSRXX', 'SSJL', 'AJJBQK', 'CPYZ', 'PJJG', 'WBWB']
        self.class_num = None
        self.shuffle = True
        self.test_size = 512
        self.max_lenth = 200
        self.min_frequency = 5
        self.batch_size = 512
        self.epoch = 20
        self.eval = 2000
        self.save = 5000
        self.max_doc_lenth = 300000
        self.embed_num = None
        self.embed_dim = 200

        self.lr = 0.001
        self.kernel_num = 128
        self.kernel_size = [3,4,5]
        self.drop_out = 0.5

        self.vocaber = None

        self.cuda = False
        self.Train = True

        self.Build_Dic = False

class Classifier():

    def __init__(self,param = clf_profile()):

        param.class_num=len(param.keys)

        self.Text = Text_processing()

        vec_model = Word2Vec.load("/media/zgy/新加卷/cuda/zhwiki/model")
        param.vec_model = vec_model
        param.vocaber = self.Text.load_vocaber()

        param.embed_num =len(param.vocaber.vocabulary_)
        print(param.embed_num)
        self.param = param


            # model = TextLSTM(param.embed_num, param.embed_dim, 100, 100, param.class_num, kernel_num=param.kernel_num,
            #                  kernel_size=param.kernel_size, vocaber=param.vocaber)
            # model.load_state_dict(torch.load("params_3050 acc 0.9408062930186823_.pkl"))
            #
            # # valid_loader = DataLoader(dataset(valid_seg_data,valid_seg_target,vocaber),batch_size=param.batch_size,num_workers=1)
            # model = model.cpu()
            # model.eval()
            # # test(model)


    def train(self):
        model = TextLSTM(self.param.embed_num, self.param.embed_dim, 100, 100, self.param.class_num, kernel_num=self.param.kernel_num,
                         kernel_size=self.param.kernel_size, vocaber=self.param.vocaber)
        # model = TextCNN(self.param.embed_num, self.param.embed_dim, 100, 100, self.param.class_num,
        #                  kernel_num=self.param.kernel_num,
        #                  kernel_size=self.param.kernel_size, vocaber=self.param.vocaber)


        # model = LSTMClassifier(300,100,param.embed_num,param.class_num,param.batch_size,True)
        print(model)
        loss_fun = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())))

        self.Train(self.Text, model, loss_fun, optimizer, self.param.epoch,
              self.param.cuda, self.param.eval, self.param.save)




    def predict(self,sentence):
        import torch.nn.functional as F
        model = TextLSTM(self.param.embed_num, self.param.embed_dim, 100, 100, self.param.class_num, kernel_num=self.param.kernel_num,
                         kernel_size=self.param.kernel_size, vocaber=self.param.vocaber)
        model.load_state_dict(torch.load("params_1450 acc 0.8994_.pkl"))
        model.eval()
        sentence = self.Text.seg(sentence)
        sentence = self.Text.remove_stop_word(sentence)
        vec = self.Text.get_embed(sentence)
        print(vec)
        ten = Variable(torch.from_numpy(np.array(vec)))
        result = model(torch.unsqueeze(ten, 0))

        result = F.softmax(result, 1)
        return result.data.numpy()[0]

        os.chdir(os.path.dirname(__file__))




    def Train(self,text,model,loss_fun,optimizer,epoch,cuda,eval,save):
        gc.collect()
        if cuda:
            model = model
        loss_total = 0
        step = 0

        test_loader = DataLoader(self.Text.re_prep_law_data_for_pytorch(train=False),batch_size=self.param.batch_size,num_workers=1)

        steps = []
        accs = []
        save_acc = open("acc_step", "wb")
        top_acc = 0
        last_file_name = None

        for ep in range(epoch):

            train_loader = DataLoader(self.Text.re_prep_law_data_for_pytorch(train=True),
                                     batch_size=self.param.batch_size, num_workers=1)

            for batch in train_loader:
                data, label = batch

                data = Variable(data[0])
                data.requires_grad=False
                label = Variable(label)
                if cuda:
                    data = data
                    label = label
                output = model(data)
                loss = loss_fun(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 10 == 0):
                    model.eval()

                    acc = self.eva(model,test_loader)
                    accs.append(acc)
                    steps.append(step)
                    if(acc>top_acc and step>0):
                        top_acc = acc
                        print(step,":new High.","Top acc:",top_acc,ep)
                        model = model.cpu()

                        if(last_file_name):
                            os.remove(last_file_name)
                        last_file_name='params_'+str(step)+" acc " +str(acc)+ '_.pkl'
                        torch.save(model.state_dict(), last_file_name)
                        model = model
                    else:
                        print(step," ",acc,".top acc is ",top_acc,ep)
                    model.train()
                if(step==0):
                    print("start grad wordvec")

                if (step % 5000 == 0):
                    pickle.dump(accs, save_acc)
                    pickle.dump(steps, save_acc)
                step += 1

            plt.plot(steps, accs)
            plt.show

        else:
            model.load_state_dict(torch.load('params_final.pkl'))


    def eva(self,model,test_loader):
        acc = 0
        total = 0
        predicts = []
        real = []
        for batch in test_loader:
            data, label = batch
            data = Variable(data[0])
            label = Variable(label)
            output = model(data)
            _, predict = torch.max(output, 1)
            predicts+=predict.data.cpu().tolist()
            real += label.data.cpu().tolist()
            acc += (predict.data.cpu() == label.data.cpu()).sum()
            total += label.data.size(0)

        m = confusion_matrix(np.array(real),np.array(predicts))
        print(m)

        return acc / total


if __name__ == "__main__":
    clf = Classifier()
    clf.train()
    # print(clf.predict("2015-12-31"))
