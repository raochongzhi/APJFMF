'''
simCSE embedding
by Ceres Rao
20230520
不用sts-b数据集进行效果测试
'''

import pandas as pd
import re
# import jieba
# from tqdm import tqdm
# from gensim.models import word2vec, Word2Vec
import random
# import time
from typing import Dict, List
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, BertForMaskedLM
from sklearn.model_selection import train_test_split
# from transformers.tokenization_utils import PreTrainedTokenizer

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(2023)

# 读取数据集
dataset = pd.read_csv('enter you filepath',dtype = {'UserID': 'str', 'JobID': 'str'})

list_job = []
for descriptions in dataset['岗位描述'].values:
    description = re.split(r'[：；。！？]', descriptions.replace('\n', ''))
    list_job.extend(description[:-1])

list_user_exp = []
for experiences in dataset['experience'].values:
    experience = re.split(r'[：；。！？]', experiences.replace('\n', ''))
    list_user_exp.extend(experience[:-1])

list_all = list_job + list_user_exp

# 0.6 0.2 0.2比例分割数据集
def train_test_val_split(list,ratio_train,ratio_test,ratio_val):
    train, middle = train_test_split(list, test_size=1-ratio_train, random_state=20)
    ratio = ratio_val/(1-ratio_train)
    test, validation = train_test_split(middle, test_size=ratio, random_state=20)
    return train, test, validation

train_data, test_data, dev_data = train_test_val_split(list_all, 0.6,0.2,0.2)

# simCSE
# 基本参数
EPOCHS = 1
SAMPLES = 10000
BATCH_SIZE = 64
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'cls'  # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预训练模型目录
model_path = 'BERT_WWM_EXT'

# 微调后参数存放位置
SAVE_PATH = 'simcse.pt'

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


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

# 损失函数
def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    sim = sim / 0.05
    loss = F.cross_entropy(sim, y_true)
    return loss

def eval(model, dataloader) -> float:
    total = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, source in enumerate(tqdm(dataloader), start=1):
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)

            pred = model(input_ids, attention_mask, token_type_ids)
            labels = torch.arange(pred.shape[0], device=DEVICE)
            pred = torch.argmax(pred, dim=1)
            correct = (labels == pred).sum()
            total_correct += correct
            total += pred.size(0)
    acc = total_correct/total
    return acc


def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)

        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                best = corrcoef
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")


print(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
tokenizer = BertTokenizer.from_pretrained(model_path)


train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE, drop_last = True)
dev_dataloader = DataLoader(TrainDataset(dev_data), batch_size=BATCH_SIZE, drop_last = True)
test_dataloader = DataLoader(TrainDataset(test_data), batch_size=BATCH_SIZE, drop_last = True)

# 定义模型
assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 模型训练
best = 0
for epoch in range(EPOCHS):
    train(model, train_dataloader, dev_dataloader, optimizer)

# 模型测试
model.load_state_dict(torch.load(SAVE_PATH))
dev_corrcoef = eval(model, dev_dataloader)
test_corrcoef = eval(model, test_dataloader)

