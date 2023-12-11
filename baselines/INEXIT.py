import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
# import os, sys
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
import time
import copy
import warnings
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
warnings.filterwarnings('ignore')
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# print(torch.cuda.is_available())

dataset = pd.read_csv('/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/dataset_user_job_all_1.csv')

# dataset = dataset.iloc[0:100, :]

job_features = ['岗位','岗位名称','岗位工作地点','岗位三级类别','岗位招聘人数', '企业上班时间','企业下班时间','企业加班情况','岗位工作经验','岗位学历要求','岗位描述']

user_features = ['简历','学校', '专业', '学历', '性别','resume']
# job_features = ['岗位描述','岗位名称','岗位工作地点','岗位三级类别','岗位招聘人数', '企业上班时间','企业下班时间','企业加班情况','岗位工作经验','岗位学历要求']
#
# user_features = ['resume','学校', '专业', '学历', '性别', '学校', '专业', '学历', '性别', '学校', '专业']

def train_test_val_split(x1, ratio_train, ratio_test, ratio_val):
    x1_train, x1_middle = train_test_split(x1, test_size=1-ratio_train, random_state=20)
    ratio = ratio_val/(ratio_test + ratio_val)
    x1_test, x1_validation = train_test_split(x1_middle, test_size=ratio, random_state=20)
    return x1_train, x1_test, x1_validation

train_dataset, test_dataset, val_dataset = train_test_val_split(dataset, 0.6, 0.2, 0.2)

# bert_path = 'bert-base-chinese'''

bert_path = '/share/home/xuaobo/ai_studio/raochongzhi/SimCSE/pretrained_model/bert_wwm_ext_chinese_pytorch'

class JobUserDataset(data.Dataset):
    '''
    Expected data shape like:(data_num, data_len)
    '''

    def __init__(self, geek, job, geek_sent, job_sent, labels):
        self.geek = geek
        self.job = job
        self.geek_sent = geek_sent
        self.job_sent = job_sent
        self.labels = labels
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path, output_hidden_states = True)
        self.max_feat_len = 16
        self.max_sent_len = 256

    def __getitem__(self, idx):
        geek_tokens = self.bert_tokenizer(self.geek[idx], padding='max_length', truncation=True,
                                          max_length=self.max_feat_len, return_tensors='pt')

        job_tokens = self.bert_tokenizer(self.job[idx], padding='max_length', truncation=True,
                                         max_length=self.max_feat_len, return_tensors='pt')

        geek_sent_tokens = self.bert_tokenizer(self.geek_sent[idx], padding='max_length', truncation=True,
                                               max_length=self.max_sent_len, return_tensors='pt')

        job_sent_tokens = self.bert_tokenizer(self.job_sent[idx], padding='max_length', truncation=True,
                                              max_length=self.max_sent_len, return_tensors='pt')

        return geek_tokens['input_ids'], geek_tokens['token_type_ids'], geek_tokens['attention_mask'], job_tokens[
            'input_ids'], job_tokens['token_type_ids'], job_tokens['attention_mask'], geek_sent_tokens['input_ids'], \
        geek_sent_tokens['token_type_ids'], geek_sent_tokens['attention_mask'], job_sent_tokens['input_ids'], \
        job_sent_tokens['token_type_ids'], job_sent_tokens['attention_mask'], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def dropout(input_tensor, dropout_prob):
  """Perform dropout.
  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `torch.nn.dropout`).
  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  x = torch.nn.dropout(1.0 - dropout_prob)
  output = x(input_tensor)
  return output

def layer_norm(input_tensor):
  """Run layer normalization on the last dimension of the tensor."""
  return torch.nn.LayerNorm(
      input_tensor, elementwise_affine=True)

def layer_norm_and_dropout(input_tensor, dropout_prob):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.net(x)
        return x

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to('cuda')
        out = self.dropout(out)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class BertMatchingModel(nn.Module):
    def __init__(self, word_emb_dim, num_heads, hidden_size, dropout, num_layers, fusion):
        super(BertMatchingModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_path, output_hidden_states = True)
        self.fusion = fusion
        for param in self.bert.parameters():
            param.requires_grad = True

        self.encoder = Encoder(word_emb_dim *2, num_heads, hidden_size, dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(num_layers)])

        self.encoder_2 = Encoder(word_emb_dim *3, num_heads, hidden_size, dropout)
        self.encoders_2 = nn.ModuleList([
            copy.deepcopy(self.encoder_2)
            for _ in range(num_layers)])

        self.geek_pool = nn.AdaptiveAvgPool2d((1, word_emb_dim))
        self.job_pool = nn.AdaptiveAvgPool2d((1, word_emb_dim))

        self.mlp = MLP(
            input_size= word_emb_dim * 3,
            output_size=1
        )

    def forward(self, geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask,
                job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask,
                geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask,
                job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask):

        # taxon_embedding
        geek_taxon = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 0, :].squeeze(1),
                                              token_type_ids=geek_tokens_token_type_ids[:, 0, :].squeeze(1),
                                              attention_mask=geek_tokens_attention_mask[:, 0, :].squeeze(
                                                  1))[0]).squeeze(
            1)  # batch_size * max_len * word_embedding_size -> batch_size * word_embedding_size
        job_taxon = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 0, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 0, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 0, :].squeeze(
                                                1))[0]).squeeze(1)

        # key_ewmbedding
        geek_key_0 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 1, :].squeeze(1),
                                              token_type_ids=geek_tokens_token_type_ids[:, 1, :].squeeze(1),
                                              attention_mask=geek_tokens_attention_mask[:, 1, :].squeeze(
                                                  1))[0]).squeeze(1)
        geek_key_1 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 2, :].squeeze(1),
                                              token_type_ids=geek_tokens_token_type_ids[:, 2, :].squeeze(1),
                                              attention_mask=geek_tokens_attention_mask[:, 2, :].squeeze(
                                                  1))[0]).squeeze(1)
        geek_key_2 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 3, :].squeeze(1),
                                              token_type_ids=geek_tokens_token_type_ids[:, 3, :].squeeze(1),
                                              attention_mask=geek_tokens_attention_mask[:, 3, :].squeeze(
                                                  1))[0]).squeeze(1)
        geek_key_3 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 4, :].squeeze(1),
                                              token_type_ids=geek_tokens_token_type_ids[:, 4, :].squeeze(1),
                                              attention_mask=geek_tokens_attention_mask[:, 4, :].squeeze(
                                                  1))[0]).squeeze(1)
        geek_key_4 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 5, :].squeeze(1),
                                              token_type_ids=geek_tokens_token_type_ids[:, 5, :].squeeze(1),
                                              attention_mask=geek_tokens_attention_mask[:, 5, :].squeeze(
                                                  1))[0]).squeeze(1)
        # geek_key_5 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 6, :].squeeze(1),
        #                                       token_type_ids=geek_tokens_token_type_ids[:, 6, :].squeeze(1),
        #                                       attention_mask=geek_tokens_attention_mask[:, 6, :].squeeze(
        #                                           1))[0]).squeeze(1)
        # geek_key_6 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 7, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 7, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 7, :].squeeze(1))[0]).squeeze(1)
        # geek_key_7 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 8, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 8, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 8, :].squeeze(1))[0]).squeeze(1)
        # geek_key_8 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 9, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 9, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 9, :].squeeze(1))[0]).squeeze(1)
        # geek_key_9 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 10, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 10, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 10, :].squeeze(1))[0]).squeeze(1)
        # geek_key_10 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 11, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 11, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 11, :].squeeze(1))[0]).squeeze(1)
        # geek_key_11 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 12, :].squeeze(1), token_type_ids=geek_tokens_token_type_ids[:, 12, :].squeeze(1), attention_mask=geek_tokens_attention_mask[:, 12, :].squeeze(1))[0]).squeeze(1)

        job_key_0 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 1, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 1, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 1, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_1 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 2, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 2, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 2, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_2 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 3, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 3, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 3, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_3 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 4, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 4, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 4, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_4 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 5, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 5, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 5, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_5 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 6, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 6, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 6, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_6 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 7, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 7, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 7, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_7 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 8, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 8, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 8, :].squeeze(
                                                1))[0]).squeeze(1)
        job_key_8 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 9, :].squeeze(1),
                                            token_type_ids=job_tokens_token_type_ids[:, 9, :].squeeze(1),
                                            attention_mask=job_tokens_attention_mask[:, 9, :].squeeze(
                                                1))[0]).squeeze(1)
        # job_key_9 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 10, :].squeeze(1),
        #                                     token_type_ids=job_tokens_token_type_ids[:, 10, :].squeeze(1),
        #                                     attention_mask=job_tokens_attention_mask[:, 10, :].squeeze(
        #                                         1))[0]).squeeze(1)
        # job_key_10 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 11, :].squeeze(1),
        #                                      token_type_ids=job_tokens_token_type_ids[:, 11, :].squeeze(1),
        #                                      attention_mask=job_tokens_attention_mask[:, 11, :].squeeze(
        #                                          1))[0]).squeeze(1)

        # value_ewmbedding
        geek_value_0 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 5, :].squeeze(1),
                                                token_type_ids=geek_tokens_token_type_ids[:, 5, :].squeeze(1),
                                                attention_mask=geek_tokens_attention_mask[:, 5, :].squeeze(
                                                    1))[0]).squeeze(1)
        geek_value_1 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 6, :].squeeze(1),
                                                token_type_ids=geek_tokens_token_type_ids[:, 6, :].squeeze(1),
                                                attention_mask=geek_tokens_attention_mask[:, 6, :].squeeze(
                                                    1))[0]).squeeze(1)
        geek_value_2 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 7, :].squeeze(1),
                                                token_type_ids=geek_tokens_token_type_ids[:, 7, :].squeeze(1),
                                                attention_mask=geek_tokens_attention_mask[:, 7, :].squeeze(
                                                    1))[0]).squeeze(1)
        geek_value_3 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 8, :].squeeze(1),
                                                token_type_ids=geek_tokens_token_type_ids[:, 8, :].squeeze(1),
                                                attention_mask=geek_tokens_attention_mask[:, 8, :].squeeze(
                                                    1))[0]).squeeze(1)
        # geek_value_4 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 14, :].squeeze(1),
        #                                         token_type_ids=geek_tokens_token_type_ids[:, 14, :].squeeze(1),
        #                                         attention_mask=geek_tokens_attention_mask[:, 14, :].squeeze(
        #                                             1))[0]).squeeze(1)
        # geek_value_5 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 15, :].squeeze(1),
        #                                         token_type_ids=geek_tokens_token_type_ids[:, 15, :].squeeze(1),
        #                                         attention_mask=geek_tokens_attention_mask[:, 15, :].squeeze(
        #                                             1))[0]).squeeze(1)
        # geek_value_6 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 16, :].squeeze(1),
        #                                         token_type_ids=geek_tokens_token_type_ids[:, 16, :].squeeze(1),
        #                                         attention_mask=geek_tokens_attention_mask[:, 16, :].squeeze(
        #                                             1))[0]).squeeze(1)
        # geek_value_7 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 17, :].squeeze(1),
        #                                         token_type_ids=geek_tokens_token_type_ids[:, 17, :].squeeze(1),
        #                                         attention_mask=geek_tokens_attention_mask[:, 17, :].squeeze(
        #                                             1))[0]).squeeze(1)
        # geek_value_8 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 18, :].squeeze(1),
        #                                         token_type_ids=geek_tokens_token_type_ids[:, 18, :].squeeze(1),
        #                                         attention_mask=geek_tokens_attention_mask[:, 18, :].squeeze(
        #                                             1))[0]).squeeze(1)
        # geek_value_9 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 19, :].squeeze(1),
        #                                         token_type_ids=geek_tokens_token_type_ids[:, 19, :].squeeze(1),
        #                                         attention_mask=geek_tokens_attention_mask[:, 19, :].squeeze(
        #                                             1))[0]).squeeze(1)
        # geek_value_10 = self.geek_pool(self.bert(input_ids=geek_tokens_input_ids[:, 20, :].squeeze(1),
        #                                          token_type_ids=geek_tokens_token_type_ids[:, 20, :].squeeze(1),
        #                                          attention_mask=geek_tokens_attention_mask[:, 20, :].squeeze(
        #                                              1))[0]).squeeze(1)

        geek_value_11 = self.geek_pool(self.bert(input_ids=geek_sent_tokens_input_ids.squeeze(1),
                                                 token_type_ids=geek_tokens_sent_token_type_ids.squeeze(1),
                                                 attention_mask=geek_sent_tokens_attention_mask.squeeze(
                                                     1))[0]).squeeze(1)

        job_value_0 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 9, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 9, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 9, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_1 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 10, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 10, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 10, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_2 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 11, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 11, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 11, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_3 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 12, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 12, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 12, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_4 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 13, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 13, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 13, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_5 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 14, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 14, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 14, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_6 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 15, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 15, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 15, :].squeeze(
                                                  1))[0]).squeeze(1)
        job_value_7 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 16, :].squeeze(1),
                                              token_type_ids=job_tokens_token_type_ids[:, 16, :].squeeze(1),
                                              attention_mask=job_tokens_attention_mask[:, 16, :].squeeze(
                                                  1))[0]).squeeze(1)
        # job_value_8 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 17, :].squeeze(1),
        #                                       token_type_ids=job_tokens_token_type_ids[:, 17, :].squeeze(1),
        #                                       attention_mask=job_tokens_attention_mask[:, 17, :].squeeze(
        #                                           1))[0]).squeeze(1)
        # job_value_9 = self.job_pool(self.bert(input_ids=job_tokens_input_ids[:, 18, :].squeeze(1),
        #                                       token_type_ids=job_tokens_token_type_ids[:, 18, :].squeeze(1),
        #                                       attention_mask=job_tokens_attention_mask[:, 18, :].squeeze(
        #                                           1))[0]).squeeze(1)
        job_value_10 = self.job_pool(self.bert(input_ids=job_sent_tokens_input_ids.squeeze(1),
                                               token_type_ids=job_sent_tokens_token_type_ids.squeeze(1),
                                               attention_mask=job_sent_tokens_attention_mask.squeeze(
                                                   1))[0]).squeeze(1)

        # Inner interaction
        if self.fusion == 'cat':
            geek_0 = torch.cat([geek_key_0, geek_value_0], dim=1).unsqueeze(1)  # batch_size * 1 * 2 word_embedding_size
            geek_1 = torch.cat([geek_key_1, geek_value_1], dim=1).unsqueeze(1)
            geek_2 = torch.cat([geek_key_2, geek_value_2], dim=1).unsqueeze(1)
            geek_3 = torch.cat([geek_key_3, geek_value_3], dim=1).unsqueeze(1)
            geek_4 = torch.cat([geek_key_4, geek_value_11], dim=1).unsqueeze(1)
            # geek_4 = torch.cat([geek_key_4, geek_value_4], dim=1).unsqueeze(1)
            # geek_5 = torch.cat([geek_key_5, geek_value_5], dim=1).unsqueeze(1)
            # geek_6 = torch.cat([geek_key_6, geek_value_6], dim=1).unsqueeze(1)
            # geek_7 = torch.cat([geek_key_7, geek_value_7], dim=1).unsqueeze(1)
            # geek_8 = torch.cat([geek_key_8, geek_value_8], dim=1).unsqueeze(1)
            # geek_9 = torch.cat([geek_key_9, geek_value_9], dim=1).unsqueeze(1)
            # geek_10 = torch.cat([geek_key_10, geek_value_10], dim=1).unsqueeze(1)
            # geek_11 = torch.cat([geek_key_11, geek_value_11], dim=1).unsqueeze(1)

            job_0 = torch.cat([job_key_0, job_value_0], dim=1).unsqueeze(1)  # batch_size * 1 * word_embedding_size
            job_1 = torch.cat([job_key_1, job_value_1], dim=1).unsqueeze(1)
            job_2 = torch.cat([job_key_2, job_value_2], dim=1).unsqueeze(1)
            job_3 = torch.cat([job_key_3, job_value_3], dim=1).unsqueeze(1)
            job_4 = torch.cat([job_key_4, job_value_4], dim=1).unsqueeze(1)
            job_5 = torch.cat([job_key_5, job_value_5], dim=1).unsqueeze(1)
            job_6 = torch.cat([job_key_6, job_value_6], dim=1).unsqueeze(1)
            job_7 = torch.cat([job_key_7, job_value_7], dim=1).unsqueeze(1)
            job_8 = torch.cat([job_key_8, job_value_10], dim=1).unsqueeze(1)

            # job_8 = torch.cat([job_key_8, job_value_8], dim=1).unsqueeze(1)
            # job_9 = torch.cat([job_key_9, job_value_9], dim=1).unsqueeze(1)
            # job_10 = torch.cat([job_key_10, job_value_10], dim=1).unsqueeze(1)
        else:
            geek_0 = (geek_key_0 + geek_value_0).unsqueeze(1)
            geek_1 = (geek_key_1 + geek_value_1).unsqueeze(1)
            geek_2 = (geek_key_2 + geek_value_2).unsqueeze(1)
            geek_3 = (geek_key_3 + geek_value_3).unsqueeze(1)
            geek_4 = (geek_key_4 + geek_value_11).unsqueeze(1)

            # geek_4 = (geek_key_4 + geek_value_4).unsqueeze(1)
            # geek_5 = (geek_key_5 + geek_value_5).unsqueeze(1)
            # geek_6 = (geek_key_6 + geek_value_6).unsqueeze(1)
            # geek_7 = (geek_key_7 + geek_value_7).unsqueeze(1)
            # geek_8 = (geek_key_8 + geek_value_8).unsqueeze(1)
            # geek_9 = (geek_key_9 + geek_value_9).unsqueeze(1)
            # geek_10 = (geek_key_10 + geek_value_10).unsqueeze(1)
            # geek_11 = (geek_key_11 + geek_value_11).unsqueeze(1)

            job_0 = (job_key_0 + job_value_0).unsqueeze(1)
            job_1 = (job_key_1 + job_value_1).unsqueeze(1)
            job_2 = (job_key_2 + job_value_2).unsqueeze(1)
            job_3 = (job_key_3 + job_value_3).unsqueeze(1)
            job_4 = (job_key_4 + job_value_4).unsqueeze(1)
            job_5 = (job_key_5 + job_value_5).unsqueeze(1)
            job_6 = (job_key_6 + job_value_6).unsqueeze(1)
            job_7 = (job_key_7 + job_value_7).unsqueeze(1)
            job_8 = (job_key_8 + job_value_10).unsqueeze(1)

            # job_8 = (job_key_8 + job_value_8).unsqueeze(1)
            # job_9 = (job_key_9 + job_value_9).unsqueeze(1)
            # job_10 = (job_key_10 + job_value_10).unsqueeze(1)
        geek = torch.cat([geek_0, geek_1, geek_2, geek_3, geek_4],
                         dim=1)  # batch_size * 12 * (2) word_embedding_size
        # , geek_5, geek_6, geek_7, geek_8, geek_9, geek_10, geek_11
        job = torch.cat([job_0, job_1, job_2, job_3, job_4, job_5, job_6, job_7, job_8],
                        dim=1)  # batch_size * 11 * (2) word_embedding_size
        # , job_9, job_10

        # print(geek.size()) #torch.Size([32, 12, 1536])
        # print(job.size()) #torch.Size([32, 12, 1536])

        for encoder in self.encoders:
            geek, job = encoder(geek), encoder(job)  # batch_size * 12 * word_embedding_size

        if self.fusion == 'cat':
            geek = torch.cat([torch.repeat_interleave(geek_taxon.unsqueeze(1), repeats=5, dim=1), geek],
                             dim=2)  # batch_size * 12 * (3) word_embedding_size
            job = torch.cat([torch.repeat_interleave(job_taxon.unsqueeze(1), repeats=9, dim=1), job],
                            dim=2)  # batch_size * 11 * (3) word_embedding_size
        else:
            geek = torch.repeat_interleave(geek_taxon.unsqueeze(1), repeats=5,
                                           dim=1) + geek  # batch_size * 5 * word_embedding_size
            job = torch.repeat_interleave(job_taxon.unsqueeze(1), repeats=9,
                                          dim=1) + job  # batch_size * 9 * word_embedding_size

        geek_job = torch.cat([geek, job], dim=1)  # batch_size * 5+9 * (3) word_embedding_size

        # print(geek_job.size()) #torch.Size([32, 14, 2304])

        for encoder_2 in self.encoders_2:
            geek_job = encoder_2(geek_job)  # batch_size * 5+9 * (3) word_embedding_size

        geek_vec, job_vec = torch.split(geek_job, (5, 9), dim=1)
        geek_vec, job_vec = self.geek_pool(geek_vec).squeeze(1), self.job_pool(job_vec).squeeze(1)
        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)

        # print(x.size()) #torch.Size([32, 2304])
        output = self.mlp(x).squeeze(1) #output: torch.Size([32, 1]) -> torch.Size([32])
        return output


def dataset_construction(dataset_type):
    geek = []
    job = []
    geek_sent = []
    job_sent = []
    labels = []
    for row in range(len(dataset_type)):
        geek.append(user_features + [str(dataset.loc[row, user_features[k]]) for k in range(1,4)])
        job.append(job_features + [str(dataset.loc[row, job_features[k]]) for k in range(1,9)])
        geek_sent.append(dataset.loc[row, user_features[5]])
        job_sent.append(dataset.loc[row, job_features[10]])
        labels.append(dataset.loc[row, 'label'])
    return geek, job, geek_sent, job_sent, labels

geek_tr, job_tr, geek_sent_tr, job_sent_tr, labels_tr = dataset_construction(train_dataset)
geek_vl, job_vl, geek_sent_vl, job_sent_vl, labels_vl = dataset_construction(val_dataset)
geek_te, job_te, geek_sent_te, job_sent_te, labels_te = dataset_construction(test_dataset)

train_datasets = JobUserDataset(geek_tr, job_tr, geek_sent_tr, job_sent_tr, labels_tr)
val_datasets = JobUserDataset(geek_vl, job_vl, geek_sent_vl, job_sent_vl, labels_vl)
test_datasets = JobUserDataset(geek_te, job_te, geek_sent_te, job_sent_te, labels_te)

torch.save(train_datasets, '/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/train.dataset')
torch.save(val_datasets, '/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/val.dataset')
torch.save(test_dataset,'/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/test.dataset')

# train_datasets = torch.load('/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/train.dataset')
# val_datasets = torch.load('/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/val.dataset')
# test_dataset = torch.load('/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/test.dataset')

batch_size = 32
# dataloader导入
train_loader = DataLoader(dataset= train_datasets, batch_size = batch_size, shuffle = True, drop_last= True)
val_loader = DataLoader(dataset = val_datasets, batch_size = batch_size, shuffle = True, drop_last= True)
test_loader = DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = True, drop_last= True)


def training(n_epoch, lr, train, valid, model, device, model_name, model_dir="./"):
    model.cuda()
    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0, last_epoch=-1)
    best_acc, best_precision, best_recall, best_f1, best_auc = 0, 0, 0, 0, 0

    for epoch in range(n_epoch):
        start_time = time.time()
        total_loss, total_acc = 0, 0
        pred_label = []
        y_label = []
        # training
        for i, (jobs_1, jobs_2, jobs_3, jobs_4, users_1, users_2, users_3, users_4, entities_1, entities_2, entities_3,
                entities_4, labels) in enumerate(train):
            # 放GPU上运行
            jobs_1 = jobs_1.to(device)
            jobs_2 = jobs_2.to(device)
            jobs_3 = jobs_3.to(device)
            jobs_4 = jobs_4.to(device)
            users_1 = users_1.to(device)
            users_2 = users_2.to(device)
            users_3 = users_3.to(device)
            users_4 = users_4.to(device)
            entities_1 = entities_1.to(device)
            entities_2 = entities_2.to(device)
            entities_2 = entities_2.to(device)
            entities_3 = entities_3.to(device)
            entities_4 = entities_4.to(device)
            # labels = labels.to(device)
            labels = labels.to(torch.float32).to(device)

            # TODO 是否考虑模型用多个优化器？
            optimizer.zero_grad()  # 将所有模型参数的梯度置为0
            outputs = model(jobs_1, jobs_2, jobs_3, jobs_4, users_1, users_2, users_3, users_4, entities_1, entities_2,
                            entities_3, entities_4)

            # print(outputs)
            #
            # print(labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_label.extend([0 if i < 0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
            y_label.extend(list(labels.cpu().detach().numpy()))
        print('[ Epoch{}: {}/{}] '.format(epoch + 1, i + 1, t_batch))
        # evaluation
        model.eval()
        with torch.no_grad():
            pred_label = []
            y_label = []
            total_loss, total_acc = 0, 0
            for i, (
            jobs_1, jobs_2, jobs_3, jobs_4, users_1, users_2, users_3, users_4, entities_1, entities_2, entities_3,
            entities_4, labels) in enumerate(valid):
                # 放GPU上运行
                jobs_1 = jobs_1.to(device)
                jobs_2 = jobs_2.to(device)
                jobs_3 = jobs_3.to(device)
                jobs_4 = jobs_4.to(device)
                users_1 = users_1.to(device)
                users_2 = users_2.to(device)
                users_3 = users_3.to(device)
                users_4 = users_4.to(device)
                entities_1 = entities_1.to(device)
                entities_2 = entities_2.to(device)
                entities_2 = entities_2.to(device)
                entities_3 = entities_3.to(device)
                entities_4 = entities_4.to(device)
                # labels = labels.to(device)
                labels = labels.to(torch.float32).to(device)

                outputs = model(jobs_1, jobs_2, jobs_3, jobs_4, users_1, users_2, users_3, users_4, entities_1,
                                entities_2, entities_3, entities_4)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                '''
                存一下预测score
                '''
                # pred_score.extend([j for j in list(outputs.cpu().detach().numpy())])
                pred_label.extend([0 if i < 0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
                y_label.extend(list(labels.cpu().detach().numpy()))
            val_losses = total_loss / v_batch
            val_acc = accuracy_score(y_label, pred_label)
            val_precision = precision_score(y_label, pred_label)
            val_recall = recall_score(y_label, pred_label)
            val_auc = roc_auc_score(y_label, pred_label)
            val_f1 = f1_score(y_label, pred_label)
            print(
                '\nVal | Loss:{:.5f} ACC:{:.5f} Precision:{:.5f} Recall:{:.5f} AUC:{:.5f} F1:{:.5f} Time:{:.6f}'.format(
                    val_losses, val_acc, val_precision, val_recall, val_auc, val_f1, time.time() - start_time))
            if val_acc > best_acc:
                best_acc = val_acc
                best_precision = val_precision
                best_recall = val_recall
                best_f1 = val_f1
                best_auc = val_auc
                torch.save(model, "{}/{}.model".format(model_dir, model_name))
                print(
                    'save model with acc: {:.3f}, recall: {:.3f}, auc: {:.3f}'.format(best_acc, best_recall, best_auc))
        print('------------------------------------------------------')
        lr_scheduler.step()
        model.train()
    return best_acc, best_precision, best_recall, best_f1, best_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch = 20
lr = 0.005
word_emb_dim = 768
num_heads = 8
hidden_size = 768
num_layers = 1
dropout = 0.7
fusion = 'cat'
model_name = 'INEXIT'
model_dir = '/share/home/xuaobo/ai_studio/raochongzhi/APJFFF/dataset/'

INEXIT_model = BertMatchingModel(word_emb_dim, num_heads, hidden_size, dropout, num_layers, fusion)

# 进行训练
best_acc, best_precision, best_recall, best_f1, best_auc = training(epoch, lr, train_loader, val_loader, INEXIT_model, device, model_name, model_dir)

# 输出结果（验证集）
print('best_acc',best_acc)
print('best_precision',best_precision)
print('best_recall',best_precision)
print('best_f1',best_f1)
print('best_auc',best_auc)

