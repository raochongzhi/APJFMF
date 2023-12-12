'''
my model APJFFF DeepFM Only for preson-job fit work
'''

# import packages
import random
import re
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
import datetime
import time
from transformers import BertConfig, BertModel, BertTokenizer, BertForMaskedLM
import warnings
warnings.filterwarnings('ignore')

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(2023)

dataset = pd.read_csv('/share/home/xuaobo/ai_studio/raochongzhi/DeepJobRec/dataset_user_job_all_test1.csv',dtype = {'JobID': 'str', 'UserID': 'str'})

# 对vec_feas进行word2vec embedding
from gensim.models import word2vec, Word2Vec

skill_job = dataset['skill_entity_en_job'].values
skill_user = dataset['skill_entity_en_user'].values

skill_job_emb = []
for skills in skill_job:
    skill_job_emb.append(skills.split(','))
dataset['skill_job'] = skill_job_emb

skill_user_emb = []
for skills in skill_user:
    skill_user_emb.append(skills.split(','))
dataset['skill_user'] = skill_user_emb

# # # 训练word2vec模型
# text_array = np.concatenate((skill_job_emb,skill_user_emb),axis=0)
# # w2v_model = word2vec.Word2Vec(text_array, size=100, window=5, min_count=2, workers=8, iter=10, sg=1)
# w2v_model = word2vec.Word2Vec(text_array, size=8, window=5, min_count=2, workers=8, iter=10, sg=1)
# w2v_model.save('word2vec_shared.model')

# 用word2vec训练
w2v_model = Word2Vec.load('/share/home/xuaobo/ai_studio/raochongzhi/DeepJobRec/word2vec_shared.model')

## TODO 后续修改，加入attention，暂时先这样
word_present_list = list(w2v_model.wv.index2word)
dataset['skill_job'] = dataset['skill_job'].apply(lambda x:[i for i in x if i in word_present_list])
dataset['skill_user'] = dataset['skill_user'].apply(lambda x:[i for i in x if i in word_present_list])
# 先用列平均值，后面看情况再改
dataset['skill_job'] = dataset['skill_job'].apply(lambda x: np.mean([np.array(w2v_model.wv[i]).reshape(1,8) for i in x], axis=0))
dataset['skill_user'] = dataset['skill_user'].apply(lambda x: np.mean([np.array(w2v_model.wv[i]).reshape(1,8) for i in x], axis=0))

skill_user_1 = []
skill_user_2 = []
skill_user_3 = []
skill_user_4 = []
skill_user_5 = []
skill_user_6 = []
skill_user_7 = []
skill_user_8 = []

skill_job_1 = []
skill_job_2 = []
skill_job_3 = []
skill_job_4 = []
skill_job_5 = []
skill_job_6 = []
skill_job_7 = []
skill_job_8 = []

for i in range(len(dataset)):
    try:
        skill_job_embedding = dataset.loc[i, 'skill_job'][0].tolist()
        skill_job_1.append(skill_job_embedding[0])
        skill_job_2.append(skill_job_embedding[1])
        skill_job_3.append(skill_job_embedding[2])
        skill_job_4.append(skill_job_embedding[3])
        skill_job_5.append(skill_job_embedding[4])
        skill_job_6.append(skill_job_embedding[5])
        skill_job_7.append(skill_job_embedding[6])
        skill_job_8.append(skill_job_embedding[7])
    except:
        skill_job_1.append(0)
        skill_job_2.append(0)
        skill_job_3.append(0)
        skill_job_4.append(0)
        skill_job_5.append(0)
        skill_job_6.append(0)
        skill_job_7.append(0)
        skill_job_8.append(0)
    skill_user_embedding = dataset.loc[i, 'skill_user'][0].tolist()
    skill_user_1.append(skill_user_embedding[0])
    skill_user_2.append(skill_user_embedding[1])
    skill_user_3.append(skill_user_embedding[2])
    skill_user_4.append(skill_user_embedding[3])
    skill_user_5.append(skill_user_embedding[4])
    skill_user_6.append(skill_user_embedding[5])
    skill_user_7.append(skill_user_embedding[6])
    skill_user_8.append(skill_user_embedding[7])


dataset['skill_user_1'] = skill_user_1
dataset['skill_user_2'] = skill_user_2
dataset['skill_user_3'] = skill_user_3
dataset['skill_user_4'] = skill_user_4
dataset['skill_user_5'] = skill_user_5
dataset['skill_user_6'] = skill_user_6
dataset['skill_user_7'] = skill_user_7
dataset['skill_user_8'] = skill_user_8

dataset['skill_job_1'] = skill_job_1
dataset['skill_job_2'] = skill_job_2
dataset['skill_job_3'] = skill_job_3
dataset['skill_job_4'] = skill_job_4
dataset['skill_job_5'] = skill_job_5
dataset['skill_job_6'] = skill_job_6
dataset['skill_job_7'] = skill_job_7
dataset['skill_job_8'] = skill_job_8

# 稠密特征
dense_feas = ['岗位薪资下限(K)','岗位薪资上限(K)','work_length','总分','岗位招聘人数']

# 文本特征
vec_feas = ['skill_job_1','skill_job_2','skill_job_3','skill_job_4',
            'skill_job_5','skill_job_6','skill_job_7','skill_job_8',
            'skill_user_1','skill_user_2','skill_user_3','skill_user_4',
            'skill_user_5','skill_user_6','skill_user_7','skill_user_8']

text_feas = ['岗位名称', '岗位描述', 'experience']

# 稀疏特征
## TODO userid jobid应该放进去吗
sparse_feas_user = ['UserID','性别','专业']
sparse_feas_job = ['JobID','企业融资阶段','企业人员规模','企业休息时间','企业加班情况','岗位一级类别','岗位三级类别','岗位工作经验','岗位招聘类型'] # '岗位二级类别'
sparse_feas_match = ['match_degree','match_loc_job','match_loc_corp']
sparse_feas = sparse_feas_user + sparse_feas_job + sparse_feas_match

def sparseFeature(feat, feat_num,embed_dim=8):
    # if len(dataset[feat].unique()) < embed_dim:
    #     embed_dim = len(dataset[feat].unique())
    return {'feat':feat, 'feat_num':feat_num, 'embed_dim':embed_dim}

def denseFeature(feat):
    return {'feat':feat}

def vecFeature(feat):
    return{'feat':feat}

## TODO 维度可以变化吗
embed_dim = 8
feature_columns = [[denseFeature(feat) for feat in dense_feas]] +[[sparseFeature(feat, len(dataset[feat].unique()), embed_dim=embed_dim) for feat in sparse_feas]]

# dataloader setting
class JobUserDataset(data.Dataset):
    '''
    Expected data shape like:(data_num, data_len)
    '''
    def __init__(self, job, user, deepfm, label):
        self.job = job
        self.user = user
        self.deepfm = deepfm
        self.label = label

    def __getitem__(self, idx):
        if self.label is None:
            return self.job[idx], self.user[idx], self.deepfm[idx]
        return self.job[idx], self.user[idx], self.deepfm[idx], self.label[idx]

    def __len__(self):
        return len(self.job)

'''
model part
'''
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

class Attention_layer(nn.Module):
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
        self.attn2 = SelfAttentionEncoder(dim) # * 2

    def forward(self, x):
        # (N, S, L, D)
        x = x.permute(1, 0, 2, 3)   # (S, N, L, D)
        # print('x_1size', x.size())
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])   # (S, N, D)
        s = x.permute(1, 0, 2)      # (N, S, D)
        c = self.biLSTM(s)[0]       # (N, S, D)
        # print('c_size', c.size())
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
        y = torch.cat([self.attn( self.W(x.permute(1, 0, 2)) + self.U( _.expand(x.shape[1], _.shape[0], _.shape[1]) ) ).permute(2, 0, 1) for _ in s ]).permute(2, 0, 1)
        # (N, D) -> (L, N, D) -> (L, N, 1) -- softmax as L --> (L, N, 1) -> (1, L, N) -> (S2, L, N) -> (N, S2, L)
        sr = torch.cat([torch.mm(y[i], _).unsqueeze(0) for i, _ in enumerate(x)])   # (N, S2, D)
        sr = torch.mean(sr, dim=1)  # (N, D)
        return sr

class CoAttention_layer(nn.Module):
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
        self.self_attn = SelfAttentionEncoder(dim) # * 2

    def forward(self, x, s):
        # (N, S1, L, D), (N, S2, D)
        x = x.permute(1, 0, 2, 3)   # (S1, N, L, D)
        sr = torch.cat([self.co_attn(_, s).unsqueeze(0) for _ in x])   # (S1, N, D)
        u = sr.permute(1, 0, 2)     # (N, S1, D)
        c = self.biLSTM(u)[0]       # (N, S1, D)
        g = self.self_attn(c)       # (N, D)
        return g

# 建立模型
'''构造模型 FM模型'''
class FM(nn.Module):
    def __init__(self, latent_dim, fea_num):
        """
        latent_dim:各个离散特征隐向量的维度
        fea_num:特征个数
        """
        super(FM, self).__init__()
        self.latent_dim = latent_dim
        # print('fea_num',fea_num)   #82
        #定义三个矩阵，一个是全局偏置，一个是一阶权重矩阵，一个是二阶交叉矩阵
        self.w0 = nn.Parameter(torch.zeros([1,]))
        self.w1 = nn.Parameter(torch.rand([fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([fea_num, latent_dim]))
    def forward(self, x):
        #x的维度是(batch_size, fea_num)
        #一阶交叉
        # x=x[:,:82]   #[32,222]
        # print("x",x.shape)
        first_order = self.w0 + torch.mm(x, self.w1) #(batch_size, 1)
        #二阶交叉
        second_order = 1/2 * torch.sum(torch.pow(torch.mm(x, self.w2), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.w2, 2)), dim=1, keepdim=True)
        return first_order + second_order

'''构造模型 DNN模型'''
class Dnn(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units:列表，每个元素表示每一层的神经单元个数，比如[256,128,64]两层网络，第一个维度是输入维度
        """
        super(Dnn, self).__init__()
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        #layer[0]: (128,64)   layer[1]:(64,32)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        for linear in self.dnn_network:
            # print("linear",linear)
            # print("x1",x.shape)  # [32,102]
            x = linear(x)
            # print("x2",x.shape)
            x = F.relu(x)
            # print("x2", x.shape)
        x = self.dropout(x)
        # print(x,x.shape)   [32,32]
        return x

'''DeepFM模型'''
class DeepFM(nn.Module):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.):
        """
        feature_columns:特征信息
        hidden_units:dnn的隐藏单元个数
        dnn_dropout:失活率
        """
        super(DeepFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        print(self.sparse_feature_cols)

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim']) for
            i, feat in enumerate(self.sparse_feature_cols)    #len=26
        })
        #  sparse_feature_cols  =   [{'feat': 'exp', 'feat_num': 27, 'embed_dim': 8},

        self.fea_num = len(self.dense_feature_cols)
        # print("len(self.dense_feature_cols)",len(self.dense_feature_cols))
        for one in self.sparse_feature_cols:
            #     # print(i)
            # print(one["embed_dim"])
            self.fea_num += one["embed_dim"]
        self.fea_num += len(vec_feas)
        # print("len(self.dense_feature_cols)",len(self.dense_feature_cols))   #=13
        # print("self.fea_num",self.fea_num)   #15*8+16+5 = 141
        hidden_units.insert(0, self.fea_num)  #在hidden_units的最前面插入self.fea_num
        # print(hidden_units)

        self.fm = FM(self.sparse_feature_cols[0]['embed_dim'], self.fea_num)
        # print("self.sparse_feature_cols[0]['embed_dim']",self.sparse_feature_cols[0]['embed_dim'])  #  =8
        self.dnn_network = Dnn(hidden_units, dnn_dropout)
        self.nn_final_linear = nn.Linear(hidden_units[-1], 1)  #[32,1]
        # print("hidden_units[-1]",hidden_units[-1])  #32   hidden_units[-1]最后一层

    def forward(self, x):
        # print(x)
        # if x in iter(dl_train):
        #     print("true")
        dense_inputs, sparse_inputs= x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):len(self.dense_feature_cols) + 15]
        vec_inputs=x[:,len(self.dense_feature_cols) + 15:]
        # print('vec_inputs',vec_inputs)
        # vec_embeds = torch.cat(vec_inputs, dim=-1)
        # print("vec_input",vec_input)
        # print(len(self.dense_feature_cols))   #2
        # print(len(self.sparse_feature_cols))  #10
        # print("sparse_inputs",sparse_inputs)   #[32,10] 10列
        sparse_inputs = sparse_inputs.long()     #将数字或字符串转换成长整型
        # print("sparse_inputs.shape[1]",sparse_inputs.shape[1])  #15
        # print(self.embed_layers['embed_' + str(2)])
        # print(sparse_inputs[:, 2])
        # print(self.embed_layers['embed_' + str(1)])
        # print(sparse_inputs[:, 1])
        # print(self.embed_layers['embed_' + str(0)])
        # print(sparse_inputs[:, 0])
        # print(self.embed_layers['embed_' + str(0)](sparse_inputs[:, 0]))
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]     #for i in range(10)   0-9
        # for i in range(0,sparse_inputs.shape[1]):
        #     print(i)
        #     sparse_embeds=self.embed_layers['embed_' + str(i)](sparse_inputs[:, i])
        #     print("sparse_embeds",sparse_embeds)
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        # print("sparse_embeds", sparse_embeds)    #[32,62]

        # 把离散特征、连续特征、文本向量 拼接作为FM和DNN的输入
        x = torch.cat([sparse_embeds, dense_inputs, vec_inputs], dim=-1)
        # Wide
        wide_outputs = self.fm(x)
        # deep
        deep_outputs = self.nn_final_linear(self.dnn_network(x))

        # 模型的最后输出
        # outputs = F.sigmoid(torch.add(wide_outputs, deep_outputs))
        outputs = torch.add(wide_outputs, deep_outputs)
        return outputs

class APJFFF_text(nn.Module):
    def __init__(self, lstm_dim, lstm_hd_size, num_lstm_layers): #, dropout
        super(APJFFF_text, self).__init__()
        '''
        APJFNN setting
        '''
        self.user_biLSTM = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_hd_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.job_biLSTM = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_hd_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )


        self.job_layer_1 = Attention_layer(lstm_hd_size * 2, lstm_hd_size)
        self.job_layer_2 = CoAttention_layer(lstm_hd_size * 2, lstm_hd_size)

        self.user_layer_1 = Attention_layer(lstm_hd_size * 2 , lstm_hd_size)
        self.user_layer_2 = CoAttention_layer(lstm_hd_size * 2 , lstm_hd_size)

        self.liner_layer = nn.Linear(lstm_hd_size * 2 * 8, 1)
        # self.mlp = MLP(
        #     input_size=lstm_hd_size * 2 * 8,
        #     output_size=1,
        #     dropout=dropout
        # )

    def forward(self, job, user):
        # print('usersize',user.size()) # torch.Size([128, 10, 768])

        # LSTM part
        user_vecs = self.user_biLSTM(user)[0].unsqueeze(2)
        # print('usersize', user_vecs.size()) # torch.Size([128, 1, 10, 128])
        job_vecs = self.job_biLSTM(job)[0].unsqueeze(2)

        # attention part
        sj, gj = self.job_layer_1(job_vecs)
        sr, gr = self.user_layer_1(user_vecs)

        # coAttention part
        gjj = self.job_layer_2(job_vecs,sr)
        grr = self.user_layer_2(user_vecs, sj)

        # concat the vectors
        x = torch.cat([gjj, grr, gjj - grr, gjj * grr, gj , gr, gj - gr, gj * gr], axis=1)
        # print('x_size1:', x.size()) # x_size1: torch.Size([128, 1024])

        # x = self.liner_layer(x)
        # fully connected layer
        # x = self.mlp(x).squeeze(1)
        # print('x_size2:',x.size()) # x_size2: torch.Size([128])
        return x

class Attention_layer_2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn1 = SelfAttentionEncoder(dim)

    def forward(self, x):
        # (N, S, L, D)
        x = x.permute(1, 0, 2, 3)   # (S, N, L, D)
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])   # (S, N, D)
        s = x.permute(1, 0, 2)      # (N, S, D)
        return s
class APJFMF_connect(nn.Module):
    def __init__(self, lstm_dim, lstm_hd_size, num_lstm_layers, vec_size_2, dropout, feature_columns, hidden_units, dnn_dropout):
        super(APJFMF_connect, self).__init__()

        self.mlp = MLP(
            input_size=1,
            output_size=1,
            dropout=dropout
        )
        self.atten_layer = Attention_layer_2(1)
        self.linear_layer = nn.Linear(2,1)
        self.text_layer = APJFFF_text(lstm_dim, lstm_hd_size, num_lstm_layers)
        self.entity_layer = DeepFM(feature_columns, hidden_units, dnn_dropout)

    def forward(self, job, user, x):
        text_vec = self.text_layer(job, user)
        # print('text_vec',text_vec)
        entity_vec = self.entity_layer(x)

        # attention
        # print('entity_vec',entity_vec)
        # vec = torch.stack([text_vec, entity_vec]).unsqueeze(0) # (N, S, L, D) -> (N, S, D)
        # vec = vec.permute(2, 1, 0, 3)
        # # print('vec_size',vec.size())
        # vec_att = self.atten_layer(vec)
        # vec_att_1 = self.linear_layer(vec_att.permute(0,2,1)).squeeze(2)
        # # print('vec_att', vec_att.size())
        # x = self.mlp(vec_att_1)
        # # print('xsize',x.size())
        # x= x.squeeze(1)

        # no attention
        vec = entity_vec
        # vec = torch.cat([text_vec,entity_vec], axis=1)
        # print('vecsize',vec.size())
        # x = self.mlp(vec).squeeze(1)
        x = F.sigmoid(vec).squeeze()
        return x

# train
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc = 0, 0
    best_acc, best_precision, best_recall, best_f1, best_auc = 0, 0, 0, 0, 0
    pred_label = []
    y_label = []

    for epoch in range(n_epoch):
        start_time = time.time()
        total_loss, total_acc = 0, 0
        # training
        for i, (jobs, users, entities, labels) in enumerate(train):

            # 放GPU上运行
            jobs = jobs.to(torch.float32)
            jobs = jobs.to(device)

            users = users.to(torch.float32)
            users = users.to(device)

            entities = entities.to(torch.float32)
            entities = entities.to(device)

            labels = labels.to(torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            # model.zero_grad()
            outputs = model(jobs, users, entities)
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
            pred_score = []
            pred_label = []
            y_label = []
            total_loss, total_acc = 0, 0
            for i, (jobs, users, entities, labels) in enumerate(valid):
                # 放GPU上运行
                jobs = jobs.to(torch.float32)
                jobs = jobs.to(device)

                users = users.to(torch.float32)
                users = users.to(device)

                entities = entities.to(torch.float32)
                entities = entities.to(device)

                labels = labels.to(torch.float32)
                labels = labels.to(device)

                outputs = model(jobs, users, entities)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                '''
                存一下预测score
                '''
                pred_score.extend([j for j in list(outputs.cpu().detach().numpy())])
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
                torch.save(model, "{}/{}20230715.model".format(model_dir, model_name))
                print('save model with acc: {:.3f}, recall: {:.3f}, auc: {:.3f}'.format(best_acc,best_recall,best_auc))
        print('------------------------------------------------------')
        model.train()
    return best_acc, best_precision, best_recall, best_f1, best_auc


# 设置模型参数
hidden_units = [128, 64, 32]
dnn_dropout = 0.
lstm_dim = 768
lstm_hd_size = 128
num_lstm_layers = 1
dropout = 0.7
vec_size_2 = 32

# 训练参数
epoch = 1000
lr = 0.005
# batch_size = 4012
batch_size = 32
model_dir = '/share/home/xuaobo/ai_studio/raochongzhi/DeepJobRec' # change with your model save path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "simcse_apjfff"


# 定义模型
APJFFF_model = APJFMF_connect(lstm_dim, lstm_hd_size, num_lstm_layers, vec_size_2, dropout, feature_columns, hidden_units, dnn_dropout)


train_dataset = torch.load("/share/home/xuaobo/ai_studio/raochongzhi/DeepJobRec/train_3.dataset")
val_dataset = torch.load("/share/home/xuaobo/ai_studio/raochongzhi/DeepJobRec/val_3.dataset")
test_dataset = torch.load("/share/home/xuaobo/ai_studio/raochongzhi/DeepJobRec/test_3.dataset")

# dataloader导入
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# 进行训练
best_acc, best_precision, best_recall, best_f1, best_auc = training(epoch, lr, train_loader, val_loader, APJFFF_model, device, model_name, model_dir)

# 输出结果
print('best_acc',best_acc)
print('best_precision',best_precision)
print('best_recall',best_precision)
print('best_f1',best_f1)
print('best_auc',best_auc)
