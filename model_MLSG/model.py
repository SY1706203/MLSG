import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, K, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.K = K
        self.emb_dimension = emb_dimension
        self.clusterCenter = torch.Tensor(emb_size, K, emb_dimension)
        self.clusterCount = torch.ones(emb_size, K)
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.sense_embeddings = nn.Embedding(emb_size * K, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.clusterCenter.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)
        self.sense_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v, rightsense,use_cuda): 
        for i in range(len(pos_u)):
            pos_u[i]=pos_u[i]*self.K+rightsense
        pos_u=Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_v = Variable(torch.LongTensor(neg_v))
        if use_cuda:
            pos_u=pos_u.cuda()
            pos_v=pos_v.cuda()
            neg_v=neg_v.cuda()
        emb_sense=self.sense_embeddings(pos_u)
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
            embedding=self.u_embeddings.weight.cpu().data.numpy()
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
            sum=self.clusterCount[wid][0]+self.clusterCount[wid][1]+self.clusterCount[wid][2]
            e0=(1.0*self.clusterCount[wid][0]/sum)*sense_emb[wid * self.K + 0]
            e1=(1.0*self.clusterCount[wid][1]/sum)*sense_emb[wid * self.K + 1]
            e2=(1.0*self.clusterCount[wid][2]/sum)*sense_emb[wid * self.K + 2]
            e=e0+e1+e2
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(100, 4, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
