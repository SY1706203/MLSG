import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

torch.manual_seed(1)


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, K, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.K = K
        self.emb_dimension = emb_dimension
        self.clusterCenter = torch.Tensor(emb_size, K, emb_dimension)
        self.clusterCount = torch.LongTensor(emb_size, K)
        self.num_sense = torch.LongTensor(emb_size)
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.sense_embeddings = nn.Embedding(emb_size * K, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        for i in range(self.emb_size):
            self.num_sense[i]=1
        for i in range(self.emb_size):
            for j in range(self.K):
                self.clusterCount[i][j]=1
        initrange = 0.5 / self.emb_dimension
        self.clusterCenter.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)
        self.sense_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v, rightsense,use_cuda):
        for i in range(len(pos_u)):
            pos_u[i] = pos_u[i] * self.K + rightsense
        pos_u = Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_v = Variable(torch.LongTensor(neg_v))
        if use_cuda:
            pos_u = pos_u.cuda()
            pos_v = pos_v.cuda()
            neg_v = neg_v.cuda()
        emb_sense = self.sense_embeddings(pos_u)
        # emb_u=self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_sense, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_sense.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save_embedding(self, id2word, file_name, sense_name,use_cuda):
        if use_cuda:
            embedding = self.v_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.v_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

        if use_cuda:
            sense_emb = self.sense_embeddings.weight.cpu().data.numpy()
        else:
            sense_emb = self.sense_embeddings.weight.data.numpy()
        fout = open(sense_name, 'w')
        fout.write("%d %d\n" % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            sum = 0
            for i in range(self.num_sense[wid]):
                sum += self.clusterCount[wid][i]
            en = np.zeros((self.num_sense[wid], self.emb_dimension))
            e = np.zeros(self.emb_dimension)
            for j in range(self.num_sense[wid]):
                en[j] = (1.0 * self.clusterCount[wid][j] / sum) * sense_emb[wid * self.K + j]
                e += en[j]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
        
        fout1=open('output/each_sense_emb.txt','w')
        fout1.write("%d %d\n" % (len(id2word), self.emb_dimension))
        for wid,w in id2word.items():
            sum = 0
            for i in range(self.num_sense[wid]):
                sum += self.clusterCount[wid][i]
            for j in range(self.num_sense[wid]):
                pro=1.0 * self.clusterCount[wid][j] / sum
                e = sense_emb[wid*self.K+j]
                e = ' '.join(map(lambda x: str(x), e))
                fout1.write(('%s %d %f %s\n') % (w,j,pro,e))
        
        fout2=open('output/max_sense_emb.txt','w')
        fout2.write("%d %d\n" % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            nummax=0
            for j in range(self.num_sense[wid]):
                if self.clusterCount[wid][j]>nummax:
                    nummax=j
            e = sense_emb[wid * self.K + nummax]
            e = ' '.join(map(lambda x: str(x), e))
            fout2.write('%s %s\n' % (w, e))

def test():
    model = SkipGramModel(100, 4, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
