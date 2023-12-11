import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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