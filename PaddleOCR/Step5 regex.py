'''
正则表达式
提取经验、个人评价等
by Ceres Rao
20230414
'''

import re
import pandas as pd

data = pd.read_csv('result_pic.csv',encoding='utf-8', dtype = {'userID': 'str'})

data.insert(2,'experience','')
data.insert(3,'evaluation','')
data.insert(4,'education','')
data.insert(5,'skill','')

def info_extract(one_data):
    # 正则表达式
    education = '教\s*育\s*\S\s*\S'

    experience = '个\s*人\s*经\s*历|社\s*会\s*经\s*历|从\s*业\s*经\s*历|工\s*作\s*经\s*历|工\s*作\s*经\s*验|实\s*践\s*经\s*历|实\s*践\s*经\s*验|' \
                 '实\s*习\s*经\s*历|实\s*习\s*经\s*验|社\s*会\s*实\s*践|实\s*习\s*实\s*训|实\s*习\s*实\s*践|校\s*园\s*经\s*历|校\s*园\s*经\s*验|' \
                 '工\s*作\s*习\s*经|在\s*校\s*经\s*历|在\s*校\s*经\s*验|项\s*目\s*经\s*验|项\s*目\s*经\s*历|在\s*校\s*项\s*目|项\s*目\s*经\s*验|' \
                 '项\s*目\s*经\s*历|校\s*内\s*实\s*践|实\s*践\s*活\s*动|科\s*研\s*经\s*历|组\s*织\s*经\s*历|校\s*园\s*活\s*动|学\s*校\s*经\s*历|' \
                 '校\s*外\s*实\s*践|项\s*目\s*背\s*景|实\s*践\s*活\s*动|研\s*究\s*经\s*历|主\s*要\s*经\s*历|社\s*会\s*活\s*动\s*经\s*历'

    skill = '专\s*业\s*技\s*能|荣\s*誉\s*证\s*书|所\s*获\s*证\s*书|证\s*书\s*奖\s*项|所\s*获\s*荣\s*誉|荣\s*誉\s*奖\s*励|相\s*关\s*荣\s*誉|' \
            '曾\s*获\s*证\s*书|曾\s*获\s*荣\s*誉|奖\s*项\s*荣\s*誉|证\s*书\s*荣\s*誉|奖\s*项\s*证\s*书|获\s*得\s*证\s*书|科\s*研\s*技\s*能|' \
            '技\s*能\s*及\s*获\s*奖|技\s*能\s*证\s*书|证\s*书\s*&\s*技\s*能|技\s*能\s*&\s*兴\s*趣|自\s*身\s*技\s*能|技\s*能\s*水\s*平|' \
            '技\s*能\s*展\s*示|证\s*书\s*/技\s*能|技\s*能\s*/证\s*书|技\s*能\s*:|技\s*能\s*：|个\s*人\s*能\s*力|获\s*奖\s*信\s*息|' \
            '相\s*关\s*技\s*能|个\s*人\s*技\s*能|获\s*奖\s*技\s*能|综\s*合\s*技\s*能|职\s*业\s*技\s*能|奖\s*励\s*技\s*能|技\s*能\s*特\s*长|' \
            '技\s*能\s*证\s*书|技\s*能\s*荣\s*誉|所\s*获\s*奖\s*励|奖\s*励\s*与\s*技\s*能|技\s*能\s*评\s*价|擅\s*长\s*技\s*能|掌\s*握\s*技\s*能|' \
            '工\s*作\s*技\s*能|证\s*书\s*及\s*获\s*奖|专\s*业\s*能\s*力|技\s*能\s*与\s*评\s*价|技\s*能\s*水\s*平|技\s*能\s*和\s*证\s*书|' \
            '获\s*奖\s*及\s*技\s*能|爱\s*好\s*与\s*技\s*能|技\s*能\s*证\s*数|电\s*脑\s*技\s*能|技\s*能\s*认\s*证'

    evaluation = '自\s*我\s*评\s*价|个\s*人\s*评\s*价|个\s*人\s*总\s*结|自\s*我\s*总\s*结|获\s*奖\s*与\s*评\s*价|自\s*我\s*描\s*述|个\s*人\s*优\s*势|' \
                 '个\s*人\s*标\s*签|自\s*评|评\s*价\s*&\s*爱\s*好|自\s*我\s*介\s*绍'

    list = [education, experience, skill, evaluation]
    index = []
    for i in range(len(list)):
        pattern = re.compile(list[i])
        search = re.search(pattern, one_data)
        if search != None:
            index.append(search.span()[0])
    index.sort()
    index.append(len(one_data) - 1)
    # 按照index列表分块
    dict = {'skill': '无', 'experience': '无', 'evaluation': '无', 'education': '无'}
    for i in range(len(index) - 1):
        part = one_data[index[i]:index[i + 1]]
        # 判断是否包含字符串:str in part
        pattern1 = re.compile(skill)
        search1 = re.search(pattern1, one_data)
        if search1 != None and part.startswith(search1[0]):
            ski_part = part
            dict['skill'] = ski_part

        pattern2 = re.compile(experience)
        search2 = re.search(pattern2, one_data)
        if search2 != None and part.startswith(search2[0]):
            exp_part = part
            dict['experience'] = exp_part

        pattern3 = re.compile(evaluation)
        search3 = re.search(pattern3, one_data)
        if search3 != None and part.startswith(search3[0]):
            eva_part = part
            dict['evaluation'] = eva_part

        pattern4 = re.compile(education)
        search4 = re.search(pattern4, one_data)
        if search4 != None and part.startswith(search4[0]):
            edu_part = part
            dict['education'] = edu_part
    return dict

for row in range(data.shape[0]):
    row_content = data.loc[row,['resume']][0]
    info_dict = info_extract(row_content)
    data.loc[row,'experience'] = info_dict['experience']
    data.loc[row,'evaluation'] = info_dict['evaluation']
    data.loc[row, 'education'] = info_dict['education']
    data.loc[row, 'skill'] = info_dict['skill']

data.to_csv('result_regex.csv',encoding='utf_8_sig', index=False)