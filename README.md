# Code for paper *Your Profile Reveals Your Traits in Talent Market: An Enhanced Person-Job Fit Representation Learning* 



## Overview

We proposed a novel **A**ttentive **P**erson-**J**ob **F**it **M**ultifaceted semantic feature **F**usion model (**APJFMF**).

![](asset/model.jpg)



## Requirements

if you want to run **the whole project**, you need to require the following environments:

```
pytorch==1.8.1+cu102
transformers==3.4.0
pandas==1.3.5
gensim==3.8.3
skilt-learn==1.0.2
jieba==0.39
transformers==3.4.0
paddleocr>=2.0.1
lanms-neo==1.0.2
PyMuPDF==1.19.0
PaddlePaddle
shapely
scikit-image
imgaug
pyclipper
lmdb
tqdm
numpy
visualdl
rapidfuzz
opencv-python
opencv-contrib-python
cython
lxml
premailer
openpyxl
attrdict
Polygon3
```

If you only need to run the **feature representation part**, you need to configure the following environment:
```
pytorch==1.8.1+cu102
transformers==3.4.0
pandas==1.3.5
gensim==3.8.3
skilt-learn==1.0.2
jieba==0.39
transformers==3.4.0
```

If you want to **structure and parse resumes**, you need to have the following environment:
```
paddleocr>=2.0.1
lanms-neo==1.0.2
PyMuPDF==1.19.0
PaddlePaddle
shapely
scikit-image
imgaug
pyclipper
lmdb
tqdm
numpy
visualdl
rapidfuzz
opencv-python
opencv-contrib-python
cython
lxml
premailer
openpyxl
attrdict
Polygon3
```


## File Structure

The following is the structure of the entire code, you can put the corresponding data into the corresponding folder to perform the required tasks.

```
dataset/
├── dataset.csv            # <resume_id, job_id, label> pairs
├── {train/valid/test}.dataset  # SimCSE and DeepFM representation
├── dataset.py             # get {train/valid/test}.dataset
model/
├── APJFMF.py              # main model of this paper
├── test.py                # model test process
├── train.py               # model train process
dataprocess/
├── negtive_sampling.py    # negtive sampling process
├── entity_processing.py   # basic process for entities
crawl/
├── qs_crawl_data.csv      # Crawl results for Chinese schools scores
PaddleOCR/
├── download_from_url.py   # download Original Resume (multi file formats)
├── word2pdf.py            # Recognition of word documents, if not recognized then to Pdf
├── pdf2img.py             # Pdf batch convert to image
├── imgOCR.py              # PaddleOCR process
├── regular.py             #regular expression
SimCSE/
├── simcse_train.py        # train SimCSE model
├── BERT/bert_wwm_ext_chinese_pytorch  # BERT chinese pretrained model
SkillNER/
├── skill_extract_job.py   # job skill entity extract
├── resume_extract_job.py  # resume skill entity extract
baselines/
├── stopwords/chinese_stopword.txt    # chinese stopwords for baselines
├── word2vec_model         # train your word2vec model and put it here
├── machine_learning.ipynb # machine laearning methods
├── JRMPM.ipynb            # JRMPM model
├── IPJF.ipynb             # IPJF model
├── INEXIT.ipynb           # INEXIT model
├── APJFNN.py              # APJFNN model
├── BPJFNN.py              # BPJFNN model
ablation/
├── APJFFF_all.ipynb       # the whole model
├── APJFMF_attention.ipynb # del attention module
├── APJFMF_coattention.ipynb # del co-attention module
├── APJFMF_simcse.ipynb    # del SimCSE module
├── back_deepfm_only.py    # del free text module
assets/
├── model.jpg              # model figure
README.md                  # explanatory document
requirements.txt           # related Libraries & Dependencies
```



## Data Privacy

The data in this paper is not available due to data privacy in the enterprise, but you can train this model on your dataset or using a publicly available dataset.

