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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0, last_epoch=-1)
    best_acc, best_precision, best_recall, best_f1, best_auc = 0, 0, 0, 0, 0

    for epoch in range(n_epoch):
        start_time = time.time()
        total_loss, total_acc = 0, 0
        pred_label = []
        y_label = []
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

            # TODO 是否考虑模型用多个优化器？
            optimizer.zero_grad()
            outputs = model(jobs, users, entities)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_label.extend([0 if i<0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
            y_label.extend(list(labels.cpu().detach().numpy()))
        print('[ Epoch{}: {}/{}] '.format(epoch+1, i+1, t_batch))

        model.eval()
        with torch.no_grad():
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
                # pred_score.extend([j for j in list(outputs.cpu().detach().numpy())])
                pred_label.extend([0 if i < 0.5 else 1 for i in list(outputs.cpu().detach().numpy())])
                y_label.extend(list(labels.cpu().detach().numpy()))
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
        lr_scheduler.step()
        model.train()
    return best_acc, best_precision, best_recall, best_f1, best_auc