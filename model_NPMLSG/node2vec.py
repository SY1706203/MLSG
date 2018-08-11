from input_data import InputData
import numpy as np
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.optim as optim
from tqdm import tqdm
import sys


class Word2Vec:
    def __init__(self,
                 output_file_name,
                 output_sense_name,
                 emb_dimension=128,
                 K=5,
                 batch_size=1,
                 window_size=5,
                 iteration=1,
                 initial_lr=0.1,
                 createClusterLambda=1.5,
                 min_count=0):
        """Initilize class parameters.
        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.
        Returns:
            None.
        """
        self.data = InputData(min_count)
        self.output_file_name = output_file_name
        self.output_sense_name = output_sense_name
        self.emb_size = len(self.data.node2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.K = K
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.createClusterLambda = createClusterLambda
        self.skip_gram_model = SkipGramModel(self.emb_size, self.K, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        total_pos_pairs = self.data.get_node_pairs(self.window_size)
        print("training\n")
        for t in process_bar:
            pos_pairs = total_pos_pairs[t]
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            # right=[]
            cnt = 0
            curword = pos_u[cnt]
            contextwords=[]
            contextwords_cuda = []
            while cnt < len(pos_u):
                contextwords.append(pos_v[cnt])
                contextwords_cuda.append(pos_v[cnt])
                cnt += 1
            contextembedding = torch.zeros(self.emb_dimension)
            contextwords_cuda=Variable(torch.LongTensor(contextwords_cuda))
            if self.use_cuda:
                contextwords_cuda=contextwords_cuda.cuda()
            emb_v = self.skip_gram_model.v_embeddings(contextwords_cuda)
            if self.use_cuda:
                emb_v_data = emb_v.cpu().data
            else:
                emb_v_data = emb_v.data
            for i in range(len(contextwords)):
                contextembedding += emb_v_data[i]
                # torch.add(contextembedding,emb_v_data[i,:],out=emb_v_data_total)
            emb_v_data_avg = contextembedding / (len(contextwords))
            # torch.div(emb_v_data_total,len(contextwords),out=emb_v_data_avg)
            minDist = np.inf
            rightsense = 0
            mu = torch.Tensor(self.emb_dimension)
            if self.skip_gram_model.num_sense[curword] == self.K:
                nC = self.K
            else:
                nC = self.skip_gram_model.num_sense[curword] + 1
            prob = torch.Tensor(nC)
            for k in range(self.skip_gram_model.num_sense[curword]):
                torch.div(self.skip_gram_model.clusterCenter[curword, k, :],
                          self.skip_gram_model.clusterCount[curword][k], out=mu)
                x_norm = torch.norm(emb_v_data_avg, p=2)
                y_norm = torch.norm(mu, p=2)
                summ = 0
                for p in range(self.emb_dimension):
                    summ += emb_v_data_avg[p] * mu[p]
                dist = 1 - summ / (x_norm * y_norm)
                prob[k] = dist
                if dist < minDist:
                    minDist = dist
                    rightsense = k
            if self.skip_gram_model.num_sense[curword] < self.K:
                if self.createClusterLambda < minDist:
                    prob[self.skip_gram_model.num_sense[curword]] = self.createClusterLambda
                    rightsense = self.skip_gram_model.num_sense[curword]
                    self.skip_gram_model.num_sense[curword] += 1
            for i in range(self.emb_dimension):
                self.skip_gram_model.clusterCenter[curword][rightsense][i] += emb_v_data_avg[i]
            self.skip_gram_model.clusterCount[curword][rightsense] += 1
            # for i in range(len(contextwords)):
            #    right.append(rightsense)

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v, rightsense,self.use_cuda)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.data[0],
                                         self.optimizer.param_groups[0]['lr']))
            if t * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * t / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(
            self.data.id2node, self.output_file_name, self.output_sense_name,self.use_cuda)


if __name__ == '__main__':
    w2v = Word2Vec(output_file_name=sys.argv[1], output_sense_name=sys.argv[2])
    w2v.train()
