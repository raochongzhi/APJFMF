# skill提取（使用skillNER包） 简历和岗位两边都进行提取

import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
import json
import requests
import pandas as pd
import time
# from zhon.hanzi import punctuation
# from zhon.hanzi import characters
import re

nlp = spacy.load('en_core_web_lg') #python -m spacy download en_core_web_lg
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# 翻译函数，word 需要翻译的内容
# 每小时1000次访问限制，超过会被封禁
def translate(word):
    # 有道词典 api
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 传输的参数，其中 i 为需要翻译的内容
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    # key 这个字典为发送给有道词典服务器的内容
    response = requests.post(url, data=key)
    # 判断服务器是否相应成功
    if response.status_code == 200:
        # 然后相应的结果
        return response.text
    else:
        print("有道词典调用失败")
        # 相应失败就返回空
        return None

# def lm_find_unchinese(file):
#     pattern = re.compile(r'[\u4e00-\u9fa5]')
#     unchinese = re.sub(pattern,"",file) #排除汉字
#     unchinese = re.sub('[{}]'.format(punctuation),"",unchinese) #排除中文符号
#     #print("unchinese:",unchinese)
#     return unchinese

# 读取数据 resume part
userdata = pd.read_csv('user_data_v1.csv',encoding='utf-8',dtype = {'userID': 'str'})
# userdata = userdata[['userID', 'skill']]
df = pd.DataFrame(columns=['UserID', 'skill_entity_en', 'skill_entity_zh'])

df = pd.read_csv('add_entity/user_skill_entity_1.csv', encoding='utf-8', dtype = {'UserID': 'str'})
print(time.time())

# 100条先跑着试试看
for i in range(14433, 16722):
    userID = userdata.loc[i,'userID']
    user_desc = userdata.loc[i,'resume'].replace('\n', '')
    if len(user_desc) > 4:
        # 翻译功能实现：
        list_trans = translate(user_desc)
        trans_result = json.loads(list_trans)['translateResult']
        # print(len(trans_result[0]))
        if len(trans_result[0]) > 1:
            job_description = ''.join(trans_result[i][0]['tgt'] for i in range(len(trans_result)))
            job_description = ''.join(re.findall(r'[A-Za-z\s]', job_description))
            # job_description = lm_find_unchinese(job_description)  # 去除中文
        else:
            job_description = ''
        annotations = skill_extractor.annotate(job_description)
        l1 = annotations['results']
        l11 = [l1['full_matches'][i]['doc_node_value'] for i in range(len(l1['full_matches']))] + [
            l1['ngram_scored'][i]['doc_node_value'] for i in range(len(l1['ngram_scored']))]
        l12 = ','.join(list(set(l11)))

        # if l12 != '':
        #     trans = translate(l12)
        #     skill_entity = json.loads(trans)['translateResult'][0][0]['tgt']
        # else:
        #     skill_entity = ''

        df = df.append(pd.DataFrame({'UserID': [userID], 'skill_entity_en': [l12]}),
                       ignore_index=True) #, 'skill_entity_zh': [skill_entity]
        print(time.time())

        if i % 250 == 0:
            time.sleep(900)

df.to_csv('user_skill_entity_2.csv',encoding='utf_8_sig',index=False)


df_1 = df[df.skill_entity_en != '']
df_2 = df_1[pd.notnull(df_1.skill_entity_en)]
df_2.to_csv('user_skill_entity_final.csv',encoding='utf_8_sig',index=False)