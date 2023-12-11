# 导入需要的包
from win32com import client
import glob
from os import path
# import pandas as pd
# import os
# import openpyxl
# import pdfplumber
# from pdfminer.pdfparser import PDFSyntaxError

# 修改成对应的文件夹位置
# jianli_path = "D:\\Personal File\\PycharmProjects\\word简历提取\\word2pdf"
jianli_word_path = 'D:\\Personal File\\PycharmProjects\\PaddleOCR-release-2.6\\jianli'
# jianli_pdf_path = 'D:\\Personal File\\PycharmProjects\\PaddleOCR-release-2.6\\jianli'

# 转换doc为pdf函数
def doc2pdf(fn):
    word = client.Dispatch("Word.Application")  # 打开word应用程序
    # for file in files:
    doc = word.Documents.Open(fn)  # 打开word文件
    doc.SaveAs("{}.pdf".format(fn[:-4]), 17)  # 另存为后缀为".pdf"的文件，其中参数17表示为pdf
    doc.Close()  # 关闭原来word文件
    word.Quit()

# 转换docx为pdf函数
def docx2pdf(fn):
    word = client.Dispatch("Word.Application")  # 打开word应用程序
    # for file in files:
    doc = word.Documents.Open(fn)  # 打开word文件
    doc.SaveAs("{}.pdf".format(fn[:-5]), 17)  # 另存为后缀为".pdf"的文件，其中参数17表示为pdf
    doc.Close()  # 关闭原来word文件
    word.Quit()

# doc\docx文件转pdf操作
for wordfile in glob.glob(jianli_word_path + "\\*"):
    filename = path.basename(wordfile)
    # print('开始转换word',filename)
    if filename[-3:] == 'doc':
        try:
            doc2pdf(jianli_word_path + "\\" + filename)
        except:
            print("文件错误:", filename)
    elif filename[-4:] == 'docx':
        try:
            docx2pdf(jianli_word_path + "\\" + filename)
        except:
            print("文件错误", filename)
