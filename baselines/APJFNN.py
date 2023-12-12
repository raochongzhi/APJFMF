import pandas as pd
import numpy as np
# import nltk
# from nltk import word_tokenize, pos_tag
# from nltk.stem import  WordNetLemmatizer
# from nltk.corpus import stopwords
# from gensim.models import Word2Vec

from tqdm import tqdm
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, classification_report

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import jieba.posseg as pseg
import jieba

from gensim.models import word2vec, Word2Vec
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset_user_job_all_1.csv')

stop_words = [line.strip() for line in open('chinese_stopword.txt',encoding='UTF-8').readlines()]

def pretreatment(comment):

    token_words = jieba.lcut(comment)
    token_words = [w for w in token_words if w not in stop_words]
    token_words =  pseg.cut(' '.join(token_words))
    cleaned_word = []
    for word, tag in token_words:
        if word.isdigit():
            continue
        else:
            cleaned_word.append(word)
    return cleaned_word

segment_job =[]
# job_set=pd.read_csv('job_information.csv')
for content in tqdm(dataset["岗位描述"].values):
#     segment.append(pretreatment(content))
    segment_job.append(list(jieba.cut(content)))
dataset["text_job"] = segment_job
# job_set.to_csv("job_set_segment.csv",index=False)

segment_user = []
# job_set=pd.read_csv('job_information.csv')
for content in tqdm(dataset["resume"].values):
#     segment.append(pretreatment(content))
    segment_user.append(list(jieba.cut(content)))
dataset["text_user"] = segment_user
# user_set.to_csv("user_set_segment.csv",index=False)

def train_word2vec(x):
    '''
    param: x is a list contain all the words
    return: the trained model
    '''

    model = word2vec.Word2Vec(x, size=200, window=5, min_count=2, workers=8,
                             iter=10, sg=1)
    return model

# w2v_model_1 = train_word2vec(dataset.text_job.values)
# w2v_model_1.save('./word2vec1.model')
w2v_model_1 = Word2Vec.load('./word2vec1.model')

# w2v_model_2 = train_word2vec(dataset.text_user.values)
# w2v_model_2.save('./word2vec2.model')
w2v_model_2 = Word2Vec.load('./word2vec2.model')

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        '''
        param: sentences: the list of corpus
               sen_len: the max length of each sentence
               w2v_path: the path storing word emnbedding model
        '''

        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word2vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        for i, word in enumerate(self.embedding.wv.vocab):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("")
        self.add_embedding("")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sentence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx[''])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        '''
        change words in sentences into idx in embedding_matrix
        '''
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx[''])
            sentence_idx = self.pad_sentence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        return torch.LongTensor(y)

# MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)        # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x

class JobLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.attn1 = SelfAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn2 = SelfAttentionEncoder(dim)

    def forward(self, x):
        # (N, S, L, D)
        x = x.permute(1, 0, 2, 3)   # (S, N, L, D)
        # print('x_1size', x.size())
        # x = self.attn1(x).unsqueeze(0)
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])   # (S, N, D)
        s = x.permute(1, 0, 2)      # (N, S, D)
        c = self.biLSTM(s)[0]       # (N, S, D)
        g = self.attn2(c)           # (N, D)
        return s, g

class CoAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=0)
        )

    def forward(self, x, s):
        # (N, L, D), (N, S2, D)
        s = s.permute(1, 0, 2)  # (S2, N, D)
        y = torch.cat([ self.attn( self.W(x.permute(1, 0, 2)) + self.U( _.expand(x.shape[1], _.shape[0], _.shape[1]) ) ).permute(2, 0, 1) for _ in s ]).permute(2, 0, 1)
        # (N, D) -> (L, N, D) -> (L, N, 1) -- softmax as L --> (L, N, 1) -> (1, L, N) -> (S2, L, N) -> (N, S2, L)
        sr = torch.cat([torch.mm(y[i], _).unsqueeze(0) for i, _ in enumerate(x)])   # (N, S2, D)
        sr = torch.mean(sr, dim=1)  # (N, D)
        return sr

class UserLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.co_attn = CoAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.self_attn = SelfAttentionEncoder(dim)

    def forward(self, x, s):
        # (N, S1, L, D), (N, S2, D)
        x = x.permute(1, 0, 2, 3)   # (S1, N, L, D)
        sr = torch.cat([self.co_attn(_, s).unsqueeze(0) for _ in x])   # (S1, N, D)
        u = sr.permute(1, 0, 2)     # (N, S1, D)
        c = self.biLSTM(u)[0]       # (N, S1, D)
        g = self.self_attn(c)       # (N, D)
        return g

class APJFNN(nn.Module):
    def __init__(self, embedding_job, embedding_user, fix_embedding=True):
        super(APJFNN, self).__init__()

        # 字典中有(行数)个词，词向量维度为(列数)
        self.user_embedding = nn.Embedding(embedding_user.size(0), embedding_user.size(1))
        self.user_embedding.weight = nn.Parameter(embedding_user)
        self.user_embedding.weight.requires_grad = False if fix_embedding else True

        self.job_embedding = nn.Embedding(embedding_job.size(0), embedding_job.size(1))
        self.job_embedding.weight = nn.Parameter(embedding_job)
        self.job_embedding.weight.requires_grad = False if fix_embedding else True

        dim = 200
        hd_size = 32
        self.user_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.job_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.job_layer = JobLayer(hd_size * 2, hd_size)
        self.user_layer = UserLayer(hd_size * 2, hd_size)

        self.mlp = MLP(
            input_size=hd_size *2 * 3,
            output_size=1,
            dropout=0.7
        )

    def forward(self, job, user):
        user = self.user_embedding(user.long())  # .long()
        job = self.job_embedding(job.long())
        # print('usersize', user.size())
        # print('jobsize:', job.size())
        user_vecs = self.user_biLSTM(user)[0].unsqueeze(1)
        job_vecs = self.job_biLSTM(job)[0].unsqueeze(1)
        # print('user_vecs_size', user_vecs.size())
        # print('job_vecs_size:', job_vecs.size())
        sj, gj = self.job_layer(job_vecs)
        gr = self.user_layer(user_vecs, sj)
        x = torch.cat([gj, gr, gj - gr], axis=1)
        x = self.mlp(x).squeeze(1)
        return x

class JobUserDataset(data.Dataset):
    def __init__(self, job, user, label):
        self.job = job
        self.user = user
        self.label = label

    def __getitem__(self, idx):
        if self.label is None:
            return self.job[idx], self.user[idx]
        return self.job[idx], self.user[idx], self.label[idx]

    def __len__(self):
        return len(self.job)

text = []
for i in dataset['text_job']:
#     print(i)
    temp = str(i[1:-1]).split(',')
    # 删除当前字符串的首尾的空格和换行符
    text.append([t.strip()[1:-1] for t in temp])
dataset['text_job'] = text

text = []
for i in dataset['text_user']:
    temp = str(i[1:-1]).split(',')
    # 删除当前字符串的首尾的空格和换行符
    text.append([t.strip()[1:-1] for t in temp])
dataset['text_user'] = text

x_t = dataset['text_job']
user_t = dataset['text_user']
y_t = dataset['label']

sen_len_user = 50
preprocess_user = Preprocess(x_t, sen_len_user, w2v_path="./word2vec1.model")
embedding1 = preprocess_user.make_embedding(load=True)
x = preprocess_user.sentence_word2idx()

sen_len = 200
preprocess = Preprocess(user_t, sen_len, w2v_path="./word2vec2.model")
embedding2 = preprocess.make_embedding(load=True)
user = preprocess.sentence_word2idx()

y = preprocess_user.labels_to_tensor(y_t)

def train_test_val_split(x1,x2,y, ratio_train, ratio_test, ratio_val):
    x1_train, x1_middle,x2_train, x2_middle,y_train, y_middle = train_test_split(x1,x2,y, test_size=1-ratio_train, random_state=20)
    ratio = ratio_val/(ratio_test + ratio_val)
    x1_test, x1_validation,x2_test, x2_validation,y_test, y_validation = train_test_split(x1_middle,x2_middle,y_middle, test_size=ratio, random_state=20)
    return x1_train, x1_test, x1_validation,x2_train, x2_test, x2_validation,y_train, y_test, y_validation

train_x, test_x,val_x,train_user,test_user,val_user,train_y,test_y,val_y=train_test_val_split(x,user,y, 0.6, 0.2, 0.2)

# dataset构建
train_dataset = JobUserDataset(train_x, train_user, train_y)
val_dataset = JobUserDataset(val_x, val_user, val_y)
test_dataset = JobUserDataset(test_x, test_user, test_y)

batch_size = 32 # 一次训练所选取的样本数
# dataset导入
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = False)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
test_loader =DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

def training(n_epoch, lr, train, valid, model, device, model_name, model_dir="./"):
    # summary model parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nstart training, total parameter:{}, trainable:{}\n".format(total, trainable))
    model.cuda()
    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-4
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0, last_epoch=-1)
    # total_loss, total_acc = 0, 0
    best_acc, best_precision, best_recall, best_f1, best_auc = 0, 0, 0, 0, 0

    for epoch in range(n_epoch):
        start_time = time.time()
        total_loss, total_acc = 0, 0
        pred_label = []
        y_label = []
        # training
        for i, (jobs, users, labels) in enumerate(train):

            # 放GPU上运行
            jobs = jobs.to(torch.float32)
            jobs = jobs.to(device)

            users = users.to(torch.float32)
            users = users.to(device)

            # entities = entities.to(torch.float32)
            # entities = entities.to(device)

            labels = labels.to(torch.float32)
            labels = labels.to(device)

            # TODO 是否考虑模型用多个优化器？
            optimizer.zero_grad() # 将所有模型参数的梯度置为0
            # model.zero_grad() # 除所有可训练的torch.Tensor的梯度
            outputs = model(jobs, users)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_label.extend([0 if i<0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
            y_label.extend(list(labels.cpu().detach().numpy()))
        train_losses = total_loss/t_batch
        train_acc = accuracy_score(y_label, pred_label)
        train_precision = precision_score(y_label, pred_label)
        train_recall = recall_score(y_label, pred_label)
        train_auc = roc_auc_score(y_label, pred_label)
        train_f1 = f1_score(y_label, pred_label)
        print('[ Epoch{}: {}/{}] '.format(epoch+1, i+1, t_batch))
        print('\nTrain | Loss:{:.5f} ACC:{:.5f} Precision:{:.5f} Recall:{:.5f} AUC:{:.5f} F1:{:.5f} Time:{:.6f}'.format(train_losses,train_acc,train_precision, train_recall,train_auc,train_f1, time.time()-start_time))

        # evaluation
        model.eval()
        with torch.no_grad():
            # pred_score = []
            pred_label = []
            y_label = []
            total_loss, total_acc = 0, 0
            for i, (jobs, users, labels) in enumerate(valid):
                # 放GPU上运行
                jobs = jobs.to(torch.float32)
                jobs = jobs.to(device)

                users = users.to(torch.float32)
                users = users.to(device)

                # entities = entities.to(torch.float32)
                # entities = entities.to(device)

                labels = labels.to(torch.float32)
                labels = labels.to(device)

                outputs = model(jobs, users)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                '''
                存一下预测score
                '''
                # pred_score.extend([j for j in list(outputs.cpu().detach().numpy())])
                pred_label.extend([0 if i < 0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
                y_label.extend(list(labels.cpu().detach().numpy()))
            # print('\nVal | Loss:{:.5f} Time:{:.6f}'.format(total_loss/v_batch, time.time()-start_time))
            val_losses = total_loss/v_batch
            val_acc = accuracy_score(y_label, pred_label)
            val_precision = precision_score(y_label, pred_label)
            val_recall = recall_score(y_label, pred_label)
            val_auc = roc_auc_score(y_label, pred_label)
            val_f1 = f1_score(y_label, pred_label)
            print('\nVal | Loss:{:.5f} ACC:{:.5f} Precision:{:.5f} Recall:{:.5f} AUC:{:.5f} F1:{:.5f} Time:{:.6f}'.format(val_losses,val_acc,val_precision, val_recall,val_auc,val_f1, time.time()-start_time))
            if val_acc > best_acc:
                best_acc = val_acc
                best_precision = val_precision
                best_recall = val_recall
                best_f1 = val_f1
                best_auc = val_auc
                torch.save(model, "{}/{}.model".format(model_dir, model_name))
                print('save model with acc: {:.3f}, recall: {:.3f}, auc: {:.3f}'.format(best_acc,best_recall,best_auc))
        print('------------------------------------------------------')
        # lr_scheduler.step()
        # 将model的模式设为train，这样optimizer就可以更新model的參數（因為刚刚转为eval模式）
        model.train()
    return best_acc, best_precision, best_recall, best_f1, best_auc

fix_embedding = False
# input_dim = train_dataset[0][1].shape[0]
model = APJFNN(embedding1, embedding2, fix_embedding=fix_embedding)
epoch = 20
lr = 0.0001
model_dir = './'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'APJFNN'

best_acc, best_precision, best_recall, best_f1, best_auc = training(epoch, lr, train_loader, val_loader, model, device, model_name, model_dir)

# 输出结果（验证集）
print('best_acc',best_acc)
print('best_precision',best_precision)
print('best_recall',best_precision)
print('best_f1',best_f1)
print('best_auc',best_auc)

def testing(model, test_loader):
    pred_label = []
    y_label = []
    model.eval()
    with torch.no_grad():
        for i, (jobs, users, labels) in enumerate(test_loader):
            # 放GPU上运行
            jobs = jobs.to(torch.float32)
            jobs = jobs.to(device)

            users = users.to(torch.float32)
            users = users.to(device)

            # entities = entities.to(torch.float32)
            # entities = entities.to(device)

            labels = labels.to(torch.float32)
            labels = labels.to(device)

            outputs = model(jobs, users)

            pred_label.extend([0 if i < 0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
            y_label.extend(list(labels.cpu().detach().numpy()))

        test_acc = accuracy_score(y_label, pred_label)
        test_precision = precision_score(y_label, pred_label)
        test_recall = recall_score(y_label, pred_label)
        test_auc = roc_auc_score(y_label, pred_label)
        test_f1 = f1_score(y_label, pred_label)
    return test_acc, test_auc, test_precision, test_recall, test_f1

# 输出结果(测试集)
test_acc, test_auc, test_precision, test_recall, test_f1 = testing(
    torch.load('APJFNN.model'), test_loader)
print('test_acc', test_acc)
print('test_precision', test_precision)
print('test_recall', test_precision)
print('test_f1', test_f1)
print('test_auc', test_auc)