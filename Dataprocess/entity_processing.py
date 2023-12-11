import pandas as pd
import numpy as np

# read pre_processed dataset
dataset = pd.read_csv('add_entity/dataset_user_job_all_2.csv',dtype = {'JobID': 'str', 'UserID': 'str'})

# 删除一些用不到的字段
del dataset['创建时间_y']
del dataset['更新时间_y']
del dataset['创建时间_x']
del dataset['更新时间_x']
del dataset['evaluation']
del dataset['education']
del dataset['skill']
del dataset['岗位关键词']
del dataset['企业简介']
del dataset['企业行业一级类别']
del dataset['企业行业二级类别']
del dataset['企业行业三级类别']
del dataset['企业ID']
del dataset['resume']
del dataset['企业福利待遇']

# 匹配向量: 学历
match_degree = []
for i in range(len(dataset)):
    job_degree = dataset.loc[i, '岗位学历要求']
    '''
        本科( 72328 ) 1
        博士 ( 1824 )
        初中及以下 ( 31 ) 1
        大专 ( 7031 ) 1
        高中 ( 268 ) 1
        硕士 ( 10162 ) 1
        学历不限 ( 1792 ) 1
        中专/中技 ( 749 ) 1
    '''
    user_degree = dataset.loc[i,'学历']
    '''
        本科( 63946 )
        博士 ( 120 )
        高中及以下 ( 108 )
        硕士 ( 27415 )
        专科 ( 2596 )
    '''

    if job_degree == '学历不限' or job_degree == '高中' or job_degree == '初中及以下':
        match_degree.append(1)
    elif job_degree == '中专/中技' or job_degree == '大专':
        if user_degree == '高中及以下':
            match_degree.append(0)
        else:
            match_degree.append(1)
    elif job_degree == '本科':
        if user_degree == '高中及以下' or user_degree == '专科':
            match_degree.append(0)
        else:
            match_degree.append(1)
    elif job_degree == '硕士':
        if user_degree == '高中及以下' or user_degree == '专科' or user_degree == '本科':
            match_degree.append(0)
        else:
            match_degree.append(1)
    elif job_degree == '博士':
        if user_degree == '高中及以下' or user_degree == '专科' or user_degree == '本科' or user_degree == '硕士':
            match_degree.append(0)
        else:
            match_degree.append(1)

dataset['match_degree'] = match_degree
del dataset['岗位学历要求']
del dataset['学历']

# 数据处理
for i in range(60835, 60841):
    dataset.loc[i, '学校'] = '西班牙内布里哈大学'

# 读取大学QS评分
rank_qs = pd.read_csv('qs_list.csv')
rank_qs = rank_qs.drop_duplicates()
# rank_qs = rank_qs.rename(columns={'学校名称1':'学校'})
dataset = pd.merge(dataset,rank_qs,how='left',on='学校')

del dataset['学校']

# 看有无遗漏
# dataset_test = dataset_1[~pd.notnull(dataset_1['省市'])]


# 查看省市匹配程度
match_loc_job = []
for j in range(len(dataset)):
    job_loc = dataset.loc[j, '岗位工作地点']
    user_loc = dataset.loc[j, '省市']
    if user_loc in job_loc:
        match_loc_job.append(1)
    else:
        match_loc_job.append(0)

dataset['match_loc_job'] = match_loc_job

match_loc_corp = []
for j in range(len(dataset)):
    corp_loc = dataset.loc[j, '企业通讯地址']
    user_loc = dataset.loc[j, '省市']
    if user_loc in corp_loc:
        match_loc_corp.append(1)
    else:
        match_loc_corp.append(0)

dataset['match_loc_corp'] = match_loc_corp

del dataset['省市']
del dataset['企业通讯地址']
del dataset['岗位工作地点']

# 对总分进行归一化操作, 归一化到 [0，100]
# arr = dataset['总分'].values
# QS_score = []
# for x in arr:
#     x = int(float(x - np.min(arr)) / (np.max(arr) - np.min(arr)) * 100)
#     QS_score.append(x)
#
# dataset['QS_score'] = QS_score
#
# del dataset['总分']

# 获取工作时长
from datetime import datetime, timedelta
work_length = []
for i in range(len(dataset)):
    start_work = dataset.loc[i, '企业上班时间']
    end_work = dataset.loc[i, '企业下班时间']
    try:
        time_format = '%H:%M'
        dt1 = datetime.strptime(start_work, time_format)
        if end_work == '24:00':
            dt2 = (datetime.strptime('00:00', time_format) + timedelta(days=1)).replace(hour=0, minute=0)
        else:
            dt2 = datetime.strptime(end_work, time_format)
        diff = dt2 - dt1
        hours = diff.total_seconds() / 3600
        work_length.append(hours)
    except:
        work_length.append(8)
dataset['work_length'] = work_length

# 删除字段
del dataset['企业上班时间']
del dataset['企业下班时间']

# 企业融资阶段处理
from sklearn.preprocessing import LabelEncoder
# le_1 = LabelEncoder()
# dataset['fin_stage'] = le_1.fit_transform(dataset['企业融资阶段'].values).tolist()
#
# del dataset['企业融资阶段']
#
# # 企业人员规模处理
# le_2 = LabelEncoder()
# dataset['corp_scale'] = le_2.fit_transform(dataset['企业人员规模'].values).tolist()
#
# del dataset['企业人员规模']
#
# # 企业休息时间
# le_3 = LabelEncoder()
# dataset['break_time'] = le_3.fit_transform(dataset['企业休息时间'].values).tolist()
# del dataset['企业休息时间']
#
# # 企业加班情况
# le_4 = LabelEncoder()
# dataset['add_work'] = le_4.fit_transform(dataset['企业加班情况'].values).tolist()
# del dataset['企业加班情况']
#
# # 岗位招聘类型
# le_5 = LabelEncoder()
# dataset['recruit_type'] = le_5.fit_transform(dataset['岗位招聘类型'].values).tolist()
# del dataset['岗位招聘类型']


dataset.to_csv('dataset_user_job_all_test1.csv',encoding='utf_8_sig',index=False)


