'''
data_processing and negtive sampling
20230520
by Ceres Rao
'''

import pandas as pd
import random
from tqdm import tqdm

# 读取文件
userdata_2 = pd.read_csv('user_data_v1.csv',encoding='utf-8',dtype = {'userID': 'str'})
userdata_2 = userdata_2.rename(columns={'userID':'UserID'})
userdata_1 = pd.read_csv('add_entity/user_skill_entity_final.csv',encoding='utf-8',dtype = {'UserID': 'str'})
userdata = pd.merge(userdata_1,userdata_2,how='left',on='UserID')
# 删除没有experience的数据（没有分析价值）
userdata = userdata[userdata.experience != '无']

jobdata_2 = pd.read_csv('job_data.csv', dtype = {'岗位ID': 'str'})
jobdata_2 = jobdata_2.rename(columns={'岗位ID':'JobID'})
jobdata_1 = pd.read_csv('add_entity/job_skill_entity_final.csv', dtype = {'JobID': 'str'})
jobdata = pd.merge(jobdata_1,jobdata_2,how='left',on='JobID')
postdata = pd.read_csv('post_data.csv', dtype = {'岗位ID': 'str', '用户ID': 'str'})
userinfo = pd.read_csv('userinfo.csv', dtype = {'用户ID': 'str'})
userinfo = userinfo.rename(columns={'用户ID':'UserID'})


# 根据user筛选userdata和postdata
user_list = userdata.UserID.unique().tolist()
post_user = postdata.用户ID.unique().tolist()
user_list = list(set(user_list) & set(user_list)) # 求交集
userdata = userdata[userdata.UserID.isin(user_list)]
postdata = postdata[postdata.用户ID.isin(user_list)]

# 根据job筛选postdata
post_job = postdata.岗位ID.unique().tolist()
job_list = jobdata.JobID.unique().tolist()
job_list = list(set(post_job) & set(job_list)) # 求交集
postdata = postdata[postdata.岗位ID.isin(job_list)]

## 负采样
# 构建字典
type_dict = {}
for i in range(0,len(jobdata)):
    if jobdata.loc[i,['JobID']][0] not in type_dict.keys():
        type_dict[jobdata.loc[i,['JobID']][0]] = jobdata.loc[i,['企业行业三级类别']][0]

# 去掉被拒绝的投递行为
post_data_positive = postdata[postdata.投递状态 != '不合适']

# 构建数据集
dataset = pd.DataFrame(columns = ["UserID","JobID","label"])
groups = post_data_positive.groupby("用户ID")
job_id = jobdata.JobID.unique().tolist()
user_ids = []
job_ids = []
labels = []

# 1:1生成负样本
for idx, group in tqdm(groups):
    size = len(group)
    exist_job = group.岗位ID.unique().tolist()

    # 对投递过岗位类别进行收集
    type_list = []
    for j in exist_job:
        type_list.append(type_dict[j])

    # 去除已投递过的岗位
    candidate_job = [i for i in job_id if i not in exist_job]

    # 去除属于投递过岗位类别的岗位
    candidate_job_del = [i for i in candidate_job if type_dict[i] not in type_list]

    # 随机生成idx
    sample_job_index = random.sample(range(0,len(candidate_job_del)),size)

    # 1:1负采样
    user_ids.extend([idx] * 2 * size)
    exist_job.extend([candidate_job_del[j] for j in sample_job_index])
    job_ids.extend(exist_job)
    label = [1] * size
    label.extend([0] * size)
    labels.extend(label)

dataset.UserID = user_ids
dataset.JobID = job_ids
dataset.label = labels

# 查看正负样本数量
dataset.drop_duplicates()
print(dataset.label.value_counts())

# 保存负采样后数据集
dataset.to_csv('add_entity/dataset_exp_skill.csv',index=False)

# dataframe拼接
dataset_1 = pd.merge(dataset,jobdata,how='left',on='JobID')
# userdata = userdata.rename(columns={'userID':'UserID'})
dataset_2 = pd.merge(dataset_1,userdata,how='left',on='UserID')

dataset_2.to_csv('add_entity/dataset_all_exp_skill.csv',index=False)

data_entity = pd.merge(dataset_2,userinfo,how='left',on='UserID')
data_entity.to_csv('add_entity/dataset_user_job_all',index=False)

data_entity_1 = data_entity.drop('skill_entity_zh',axis=1)
data_entity_1.to_csv('add_entity/dataset_user_job_all_1.csv',index=False)
# jobid_list = list(set(data_entity_1.JobID.tolist()))

data_entity_2 = data_entity_1.rename(columns={'skill_entity_en':'skill_entity_en_user'})
data_entity_2.to_csv('add_entity/dataset_user_job_all_2.csv',index=False)

