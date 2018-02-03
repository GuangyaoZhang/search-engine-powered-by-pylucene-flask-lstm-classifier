from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class grad():
    grad = True

class TextLSTM(nn.Module):

    def __init__(self, vocab_size,embed_dim,hidden_size,linear_hidden_size,num_classes,kernel_num,kernel_size,vocaber):

        super(TextLSTM, self).__init__()
        self.vocaber = vocaber
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.convs = nn.ModuleList([nn.Conv2d(1,kernel_num,kernel_size=(k,self.hidden_size)) for k in kernel_size])

        # self.conv = nn.Conv2d(1,128,kernel_size=(3,self.hidden_size))


        self.linears = nn.Sequential(
            nn.Linear(self.hidden_size, self.linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.linear_hidden_size, self.num_classes),
            # nn.Softmax()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(kernel_num*len(kernel_size), self.linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.linear_hidden_size, self.num_classes),
            # nn.Softmax()
        )

        # self.init_embbedings()

    def init_embbedings(self):
        from gensim.models import Word2Vec
        import numpy as np
        model = Word2Vec.load('/media/zgy/新加卷/cuda/zhwiki/model')
        ran_ukn = np.random.randn(self.embedding.weight.size(1))
        for i in range(self.embedding.weight.size(0)):
            try:
                word =list(self.vocaber.reverse([[i]]))[0]
                array = np.array(model.wv[word])
                self.embedding.weight[i].data.copy_(torch.from_numpy(array))
            except KeyError:

                self.embedding.weight[i].data.copy_(torch.from_numpy(ran_ukn))
        # self.embedding.weight.detach_()
    def forward(self, x):
        x = self.embedding(x)

        if(not grad.grad):
            x.detach_()

        lstm_out, _ = self.lstm(x)
        conv_outs = [conv(lstm_out.unsqueeze(1)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(conv_out,kernel_size=(conv_out.size(2))).squeeze(2) for conv_out in conv_outs]
        out = torch.cat(out,1)
        out = self.linear2(out)
        return out

        # x = self.embedding(x)
        # lstm_out, _ = self.lstm(x)
        # out = self.linears(lstm_out[:, -1, :])
        # return out

class TextCNN(nn.Module):

    def __init__(self, vocab_size,embed_dim,hidden_size,linear_hidden_size,num_classes,kernel_num,kernel_size,vocaber):

        super(TextCNN, self).__init__()
        self.vocaber = vocaber
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.num_classes = num_classes

        self.embeded = nn.Embedding(self.vocab_size, self.embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, embed_dim)) for k in kernel_size])

        self.dropout = nn.Dropout(0.5)
        self.le = nn.Linear(len(kernel_size) * kernel_num, 128)
        self.le2 = nn.Linear(128, num_classes)

        # self.init_embbedings()


    def forward(self, x):
        x = self.embeded(x)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.le(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.le2(x)
        return x

        # x = self.embedding(x)
        # lstm_out, _ = self.lstm(x)
        # out = self.linears(lstm_out[:, -1, :])
        # return out


