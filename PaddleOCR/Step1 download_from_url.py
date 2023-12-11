import openpyxl
import requests
import pandas as pd

data = pd.read_excel(r'D:\Personal File\PycharmProjects\PaddleOCR-release-2.6\用户简历数据2212090858.xlsx')
# data1 = pd.read_excel(r'D:\Personal File\PycharmProjects\PaddleOCR-release-2.6\用户投递行为数据2212090858.xlsx')
# data1.to_csv('post_result.csv', encoding='utf_8_sig',index=False)
# 存放简历的地址
path = 'D:/Personal File/PycharmProjects/PaddleOCR-release-2.6/jianli'

# wb = openpyxl.load_workbook(r'D:\Personal File\PycharmProjects\jiuyeProject\用户简历数据.xlsx')  # 输入存放url链接的Excel电脑路径，可以修改
# sheet = wb['Sheet1']  # excel的sheet页，可以修改

for i in range(0, len(data)):  # excel中数据的行数
    name = data.loc[i, ['用户ID']][0]
    url = data.loc[i, ['附件简历']][0]
    # name = sheet['A' + str(i + 1)].value  ##此为PDF的命名，名字在表中A列
    # url = sheet['B' + str(i + 1)].value  ##PDF链接在表中B列，根据实际情况做更改
    if url[-3:] == 'pdf':
        file = open(path + '/' + str('%d'%name) + '.pdf', 'wb')
    elif url[-3:] == 'doc':
        file = open(path + '/' + str('%d'%name) + '.doc', 'wb')
    elif url[-4:] == 'docx':
        file = open(path + '/' + str('%d'%name) + '.docx', 'wb')
    elif url[-4:] == 'xlsx':
        file = open(path + '/' + str('%d'%name) + '.xlsx', 'wb')
    elif url[-3:] == 'png':
        file = open(path + '/' + str('%d'% name) + '.png', 'wb')
    elif url[-4:] == 'pptx':
        file = open(path + '/' + str('%d'% name) + '.pptx', 'wb')
    elif url[-3:] == 'wps':
        file = open(path + '/' + str('%d'% name) + '.wps', 'wb')
    elif url[-3:] == 'zip':
        file = open(path + '/' + str('%d'% name) + '.zip', 'wb')
    elif url[-3:] == 'rar':
        file = open(path + '/' + str('%d'% name) + '.rar', 'wb')
    else:
        file = open(path + '/else by code/' + str('%d'%name) + '.pdf', 'wb')
        print(str('%d'%name),str(url))
    res = requests.get(url)
    # with open(file_path, 'wb') as file:
    for chunk in res.iter_content(100000):
        file.write(chunk)
    file.close()