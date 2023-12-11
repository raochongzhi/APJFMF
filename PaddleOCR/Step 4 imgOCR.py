from paddleocr import PaddleOCR
import os
import pandas as pd

# os.environ['NUMEXPR_MAX_THREADS'] = '32'

# 图片地址
pic_all_path_test = r'D:\Personal File\PycharmProjects\PaddleOCR-release-2.6\jianli_img'

# paddleOCR使用
def imgOCR(pic_path):
    ocr = PaddleOCR(lang='ch')
    result = ocr.ocr(pic_path)
    str = ''
    for line in result:
        for i in line:
            str += i[1][0] + ' '
    return str

# 读取每一个文件夹的图片
def dir_concat(dirpath_list, filename_list):
    ocr_result = ''
    for filename in filename_list:
        file_path = os.path.join(dirpath_list, filename)
        ocr_result += imgOCR(file_path)
    return ocr_result

# 批量识别，并写入dataframe
def imgALL(pic_all_path):
    df = pd.DataFrame(columns=['userID', 'resume'])
    for dirpath, dirnames, filenames in os.walk(pic_all_path):
        # 第一个总地址不读取
        if dirpath == 'D:\Personal File\PycharmProjects\PaddleOCR-release-2.6\jianli_img':
            print('开始进行简历信息提取')
        else:
            print('识别用户ID为：', dirpath[-19:])
            result = dir_concat(dirpath, filenames)
            print('识别结果为：', result)
            df = df.append(pd.DataFrame({'userID': [dirpath[-19:]], 'resume': [result]}), ignore_index=True)
    return df

# 调用函数
result_final = imgALL(pic_all_path_test)

result_final.to_csv('result_pic.csv',encoding='utf_8_sig', index=False)
result_final.to_excel('result_pic.xlsx',encoding='utf_8_sig', index=False)
