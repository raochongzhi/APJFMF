import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')



filepath = 'your dataset root'
dataset = pd.read_csv(filepath,dtype = {'JobID': 'str', 'UserID': 'str', 'label': 'str'})

# 稠密特征
dense_feas = ['岗位薪资下限(K)','岗位薪资上限(K)','work_length','总分','岗位招聘人数']
# 文本特征
vec_feas = ['skill_job_1','skill_job_2','skill_job_3','skill_job_4',
            'skill_job_5','skill_job_6','skill_job_7','skill_job_8',
            'skill_user_1','skill_user_2','skill_user_3','skill_user_4',
            'skill_user_5','skill_user_6','skill_user_7','skill_user_8']
text_feas = ['岗位名称', '岗位描述', 'experience']
# 稀疏特征
sparse_feas_user = ['UserID','性别','专业']
sparse_feas_job = ['JobID','企业融资阶段','企业人员规模','企业休息时间','企业加班情况','岗位一级类别','岗位三级类别','岗位工作经验','岗位招聘类型'] # '岗位二级类别'
sparse_feas_match = ['match_degree','match_loc_job','match_loc_corp']
sparse_feas = sparse_feas_user + sparse_feas_job + sparse_feas_match

def sparseFeature(feat, feat_num, embed_dim=8):
    # if len(dataset[feat].unique()) < embed_dim:
    #     embed_dim = len(dataset[feat].unique())
    return {'feat':feat, 'feat_num':feat_num, 'embed_dim':embed_dim}

def denseFeature(feat):
    return {'feat':feat}

def vecFeature(feat):
    return{'feat':feat}

embed_dim = 8
feature_columns = ([[denseFeature(feat) for feat in dense_feas]]
                   +[[sparseFeature(feat, len(dataset[feat].unique()),
                                    embed_dim=embed_dim) for feat in sparse_feas]])

class FM(nn.Module):
    def __init__(self, latent_dim, fea_num):
        """
        latent_dim:各个离散特征隐向量的维度
        fea_num:特征个数
        """
        super(FM, self).__init__()
        self.latent_dim = latent_dim
        self.w0 = nn.Parameter(torch.zeros([1,]))
        self.w1 = nn.Parameter(torch.rand([fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([fea_num, latent_dim]))
    def forward(self, x):
        first_order = self.w0 + torch.mm(x, self.w1)
        second_order = 1/2 * torch.sum(torch.pow(torch.mm(x, self.w2), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.w2, 2)), dim=1, keepdim=True)
        return first_order + second_order

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

class APJFFF_entity(nn.Module):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.):
        """
        feature_columns:特征信息
        hidden_units:dnn的隐藏单元个数
        dnn_dropout:失活率
        """
        super(APJFFF_entity, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        print(self.sparse_feature_cols)

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim']) for
            i, feat in enumerate(self.sparse_feature_cols)    #len=26
        })

        self.fea_num = len(self.dense_feature_cols)
        for one in self.sparse_feature_cols:
            self.fea_num += one["embed_dim"]
        self.fea_num += len(vec_feas)
        hidden_units.insert(0, self.fea_num)  #在hidden_units的最前面插入self.fea_num

        self.fm = FM(self.sparse_feature_cols[0]['embed_dim'], self.fea_num)
        self.dnn_network = Dnn(hidden_units, dnn_dropout)
        self.nn_final_linear = nn.Linear(hidden_units[-1], 1)  #[32,1]

    def forward(self, x):
        dense_inputs, sparse_inputs= x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):len(self.dense_feature_cols) + 15]
        vec_inputs=x[:,len(self.dense_feature_cols) + 15:]
        sparse_inputs = sparse_inputs.long()     #将数字或字符串转换成长整型
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]     #for i in range(10)   0-9
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        x = torch.cat([sparse_embeds, dense_inputs, vec_inputs], dim=-1)
        # Wide
        wide_outputs = self.fm(x)
        # deep
        deep_outputs = self.nn_final_linear(self.dnn_network(x))
        outputs = torch.add(wide_outputs, deep_outputs)
        return outputs

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
    def __init__(self, dim):
        super().__init__()
        self.attn = SelfAttentionEncoder(dim) # * 2

    def forward(self, x):
        g = self.attn(x)
        return g

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
        s = s.permute(1, 0, 2)
        y = torch.cat([self.attn(self.W(x.permute(1, 0, 2)) + self.U( _.expand(x.shape[1], _.shape[0], _.shape[1]) ) ).permute(2, 0, 1) for _ in s ]).permute(2, 0, 1)
        sr = torch.cat([torch.mm(y[i], _).unsqueeze(0) for i, _ in enumerate(x)])
        sr = torch.sum(sr, dim=1)
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
        s = s.unsqueeze(2)
        s = s.permute(1, 0, 2, 3)
        sr = torch.cat([self.co_attn(x, _).unsqueeze(0) for _ in s])
        c = sr.permute(1, 0, 2)
        g = self.self_attn(c)
        return g


class APJFFF_text(nn.Module):
    def __init__(self, lstm_dim, lstm_hd_size, num_lstm_layers, dropout):  # , dropout
        super(APJFFF_text, self).__init__()
        '''
        APJFFF setting
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

        self.job_layer_1 = Attention_layer(lstm_hd_size * 2)  # , lstm_hd_size
        self.job_layer_2 = CoAttention_layer(lstm_hd_size * 2, lstm_hd_size)

        self.user_layer_1 = Attention_layer(lstm_hd_size * 2)  # , lstm_hd_size
        self.user_layer_2 = CoAttention_layer(lstm_hd_size * 2, lstm_hd_size)

        self.mlp = MLP(
            input_size=lstm_hd_size * 2 * 8,
            output_size=1,
            dropout=dropout
        )
        self.liner_layer = nn.Linear(lstm_hd_size * 2 * 8, 1)

    def forward(self, job, user):
        # LSTM part
        user_vecs = self.user_biLSTM(user)[0]
        job_vecs = self.job_biLSTM(job)[0]

        # attention part
        gj = self.job_layer_1(job_vecs)
        gr = self.user_layer_1(user_vecs)

        # coAttention part
        gjj = self.job_layer_2(job_vecs, user_vecs)
        grr = self.user_layer_2(user_vecs, job_vecs)

        # concat the vectors
        x = torch.cat([gjj, grr, gjj - grr, gjj * grr, gj, gr, gj - gr, gj * gr], axis=1)
        x = self.mlp(x)
        return x

class APJFF_connect(nn.Module):
    def __init__(self, lstm_dim, lstm_hd_size, num_lstm_layers, vec_size_2, dropout, feature_columns, hidden_units, dnn_dropout):
        super(APJFF_connect, self).__init__()

        self.mlp = MLP(
            input_size=1,
            output_size=1,
            dropout= 0.7
        )
        self.atten_layer = Attention_layer(1)
        self.linear_layer = nn.Linear(2,1)
        self.text_layer = APJFFF_text(lstm_dim, lstm_hd_size, num_lstm_layers, dropout)
        self.entity_layer = APJFFF_entity(feature_columns, hidden_units, dnn_dropout)

    def forward(self, job, user, x):
        text_vec = self.text_layer(job, user)
        entity_vec = self.entity_layer(x)

        # attention
        vec = torch.stack([text_vec, entity_vec]) # (N, S, D) -> (N, D)
        vec = vec.permute(1, 0, 2) # 2, 1, 0, 3
        vec_att = self.atten_layer(vec)
        x = self.mlp(vec_att)
        x = x.squeeze(1)
        return x



# 设置模型参数
hidden_units = [128, 64, 32]
dnn_dropout = 0.7


lstm_dim = 768
lstm_hd_size = 128
num_lstm_layers = 1
dropout = 0.7
vec_size_2 = 32

# 训练参数
epoch = 200
lr = 0.005
# batch_size = 4012
batch_size = 32
model_dir = '/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/' # change with your model save path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "APJFFF"

# 定义模型
APJFFF_model = APJFF_connect(lstm_dim, lstm_hd_size, num_lstm_layers, vec_size_2, dropout, feature_columns, hidden_units, dnn_dropout)

train_dataset = torch.load("/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/train_4.dataset")
val_dataset = torch.load("/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/val_4.dataset")
test_dataset = torch.load("/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/test_4.dataset")

# dataloader导入
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# model_path = '/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/APJFFF.model'
# APJFFF_model = torch.load(model_path) #, map_location=device

# 进行训练
best_acc, best_precision, best_recall, best_f1, best_auc = training(epoch, lr, train_loader, val_loader, APJFFF_model, device, model_name, model_dir)

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
        for i, (jobs, users, entities, labels) in enumerate(test_loader):
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
    torch.load('/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/APJFFF.model'), test_loader)
print('test_acc', test_acc)
print('test_precision', test_precision)
print('test_recall', test_precision)
print('test_f1', test_f1)
print('test_auc', test_auc)