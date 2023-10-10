
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
from sklearn.metrics import roc_auc_score, accuracy_score,recall_score,f1_score
import datetime
from transformers import BertConfig, BertModel, BertTokenizer, BertForMaskedLM
import warnings
import random
warnings.filterwarnings('ignore')
from gensim.models import word2vec, Word2Vec

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(2023)

dataset = pd.read_csv('filepath',dtype = {'JobID': 'str', 'UserID': 'str'})

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

embed_dim = 8
feature_columns = [[denseFeature(feat) for feat in dense_feas]] +[[sparseFeature(feat, len(dataset[feat].unique()), embed_dim=embed_dim) for feat in sparse_feas]]

for feat in sparse_feas:
    le = LabelEncoder()
    dataset[feat]=dataset[feat].astype('str')
    dataset[feat] = le.fit_transform(dataset[feat])

mms = MinMaxScaler()
dataset[dense_feas] = mms.fit_transform(dataset[dense_feas])

def train_test_val_split(x1, ratio_train, ratio_test, ratio_val):
    x1_train, x1_middle = train_test_split(x1, test_size=1-ratio_train, random_state=20)
    ratio = ratio_val/(ratio_test + ratio_val)
    x1_test, x1_validation = train_test_split(x1_middle, test_size=ratio, random_state=20)
    return x1_train, x1_test, x1_validation

# 划分训练集、测试集和验证集
train_dataset, test_dataset, val_dataset = train_test_val_split(dataset, 0.6, 0.2, 0.2)

# 划分之后取DeepFM部分用于后续训练
train_dataset_DeepFM = train_dataset[dense_feas + sparse_feas + vec_feas + ['label']]
test_dataset_DeepFM = test_dataset[dense_feas + sparse_feas + vec_feas + ['label']]
val_dataset_DeepFM = val_dataset[dense_feas + sparse_feas + vec_feas + ['label']]

trn_x_DeepFM = train_dataset_DeepFM.drop('label', axis=1).values
trn_y_DeepFM = train_dataset_DeepFM['label'].values

val_x_DeepFM = val_dataset_DeepFM.drop('label', axis=1).values
val_y_DeepFM = val_dataset_DeepFM['label'].values

test_x_DeepFM = test_dataset_DeepFM.drop('label', axis=1).values
test_y_DeepFM = test_dataset_DeepFM['label'].values

train_x = torch.FloatTensor(trn_x_DeepFM)
val_x = torch.FloatTensor(val_x_DeepFM)
test_x = torch.FloatTensor(test_x_DeepFM)

EPOCHS = 1
SAMPLES = 10000
BATCH_SIZE = 64
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'cls'  # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'bert_model_path'
SAVE_PATH = 'simcse_model_path'
tokenizer = BertTokenizer.from_pretrained(model_path)

def job_embedding(datasets):
    job_emb = []
    for descriptions in datasets['岗位描述'].values:
        description = re.split(r'[：；。！？]', descriptions.replace('\n', ''))
        # print(len(description))
        if len(description) > 10:
            description = description[:10]
        else:
            pad_length = 10 - len(description)
            for i in range(pad_length):
                description.append('')
        job_emb.append(description)
    return job_emb


# 经验部分padding 1864828
def user_embedding(datasets):
    user_emb = []
    for experiences in datasets.experience.values:
        experience = re.split(r'[：；。！？]', experiences.replace('\n', ''))
        if len(experience) > 10:
            experience = experience[:10]
        else:
            pad_length = 10 - len(experience)
            for i in range(pad_length):
                experience.append('')
        user_emb.append(experience)
    return user_emb

train_job = job_embedding(train_dataset)
train_user = user_embedding(train_dataset)

test_job = job_embedding(test_dataset)
test_user = user_embedding(test_dataset)

val_job = job_embedding(val_dataset)
val_user = user_embedding(val_dataset)

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

class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, pooler_output, hidden_states = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return last_hidden_state[:, 0]

        if self.pooling == 'pooler':
            return pooler_output

        if self.pooling == 'last-avg':
            last = last_hidden_state
            return ((last * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

        if self.pooling == 'first-last-avg':
            first = hidden_states[1]
            last = hidden_states[-1]
            pooled_result = ((first + last) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                -1).unsqueeze(-1)
            return pooled_result

# simCSE预测向量表示
def predict(model,text):
    # model.cuda()
    model.eval()
    with torch.no_grad():
        text_input_ids = text.get('input_ids').squeeze(1).to(DEVICE)
        text_attention_mask = text.get('attention_mask').squeeze(1).to(DEVICE)
        text_token_type_ids = text.get('token_type_ids').squeeze(1).to(DEVICE)
        text_pred = model(text_input_ids, text_attention_mask, text_token_type_ids)
    return text_pred

# text转id
def text_2_id(text):
    return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')

# simCSE 模型定义
assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
simcse_model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)
simcse_model.load_state_dict(torch.load(SAVE_PATH))

# 分别转tensor
def txt2tensor(texts):
     tensor3d = torch.stack([torch.cat([predict(simcse_model, text_2_id(text[i])) for i in range(10)]) for text in texts],dim = 0)
     return tensor3d

train_job_tensor = txt2tensor(train_job)
train_user_tensor = txt2tensor(train_user)
train_y_tensor = torch.LongTensor(trn_y_DeepFM)

val_job_tensor = txt2tensor(val_job)
val_user_tensor = txt2tensor(val_user)
val_y_tensor = torch.LongTensor(val_y_DeepFM)

test_job_tensor = txt2tensor(test_job)
test_user_tensor = txt2tensor(test_user)
test_y_tensor = torch.LongTensor(test_y_DeepFM)

train_dataset = JobUserDataset(train_job_tensor, train_user_tensor, train_x, train_y_tensor)
val_dataset = JobUserDataset(val_job_tensor, val_user_tensor, val_x, val_y_tensor)
test_dataset = JobUserDataset(test_job_tensor, test_user_tensor, test_x, test_y_tensor)
