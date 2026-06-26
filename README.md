# Title: ML Model of Optimizing SOC through Big Data Analysis, DM and ML
# Description
This python code is a result of our research in optimizing Security Operations Centre through a combination of Big Data Analysis incorporating data mining and machine learning techniques. 
The code is a practical execution of an ensemble of DM/ ML algorithms on Microsoft GUIDE, the SOC specific dataset released by Microsoft in 2025. 
The code is designed to achieve a remarkable reduction in false positives of SOC while maintaining a very good level of true positive identifications.
the code carries out these steps successfully in an automated manner when provided the required dataset and run within the python IDE
The relevant research for this code has been written, is aimed at laying the foundation stone for achieveing a fully automated, self driven contextually aware SOC one day.

# Dataset Information 
The dataset utilized for executing the code upon is Microsoft's GUIDE dataset, the SOC specific dataset released in 2025 for public research. 
The original dataset of Microsoft GUIDE consisted of raw telemetry data related to SOC, consisting of 46 features/ columns, which can be divided into 5 main cateogries of information as follows: -
1. Identifiers
2. Alert Meta Data
3. MITRE ATT&CK annotations
4. Host/ network context
5. Enrichment Labels

The original dataset consists of 7,95,426 rows of records. 
Four sequential processes have been executed on the given dataset to achieve following: -
1. Removal of attributes that do not affect alert correctness or contribute to inferring the type of an offence (i.e. TP/FP)
2. Removal of leakage attirbutes - i.e attributes which may cause a bias within the learning process of the applied ML algorithm
3. Removal of redundancy and noise through feature engineering
4. Normalization and binarization of values
5. Label trasnformation
6. Encoding of important features for utilization by ML algorithms

The resulting code contains the same number of records/ rows as the original one but after the execution of feature engineering and encoding processes the new dataset contains 790 columns

The steps carried out are explained under the heading of methodology

# Code Information 
The code file provided within this repository is named "SOC_optimization.py". it s python file that can be run within any python related IDE.
However we have utilized pycharm for the said purpose. The code contains following execution steps
1. importing of libraries
2. loading of data
3. creating binary levels
4. removing leakage columns from the dataset
5. feature engineering steps
6. splitting features and labels
7. categorical encoding of features
8. export of preprocessed dataset (optional)
9. train-test split
10. execution of random forest algorithm
11. execution of XGboost algorithm
12. execution of isolation forest for anomaly detection
13. fusion of models
14. evaluation of results
15. identifying important features
16. generating graphs of confusion matrix and feature importance
17. execution of shuffle sanity test
    
# Method of Access, Usage Instructions and Loading of Data
the subject dataset i.e Microsoft GUIDE can be downloaded from 
The subject Data sub set is in CSV format with a size of 3.48 GB and having 796 columns. 
Since such a large dataset cannot be uploaded directly therefore it has been placed on google drive and a link has been placed within this repository. 
The access is open for this database with a view to facilitate researches across the globe in making their efforts for achieving a better, more optimized and automated SOC. 

## Method of Access - Dataset Link
The pre-processed dataset can be accessed and downloaded through this link: https://drive.google.com/file/d/1lOPmmG2cnZX3ORvevhEH1Ypo5jxCnIrJ/view?usp=drive_link
The downloaded version of dataset has large number of fields and therefore cannot be previewed by normal worksheet softwares such as excel. It is recommended to download it and utilize within the python script. 
Moreover the zipped version of the dataset is also available on the link https://drive.google.com/file/d/1rKx3uOlaeg_nVVVAm1Xm6Q7gaRBSOrr6/view?usp=drive_link 

## Loading of Data
Any researcher who desires to utilize this pre-processed dataset version of Microsoft GUIDE, may download it to his specific folder, from hwere he can upload it into any of his python program by adding the following command within his script :-

DATA_PATH = "/(path to the folder location where the downloaded dataset is stored)/processed_GUIDE.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print(df['IncidentGrade'].value_counts())

## Usage Instructions
The aspirant researchers may ingest this pre-processed dataset within their python programms and execute their designed ML algorithms on it directly, without having to go through the hassle of exectuing feature engineering processes on it again. 

# Requirements
In order to upload and tulize the data within python scripts, the specific python libraries requried can be called wtihin the script as follows: -

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

# Methodology 
The processes executed on the original Microsfot GUIDE Dataset to achieve its pre-processed, Machine Learning ready version, are enumerated as follows: -

1. Column normailzation after dataset loading
2. Binary Label construction
3. Leakage attirbutes removal
4. Irrelevant attributes removal
5. Feature engineering
   a. Temporal feature engineering - transforming raw timestamp into time of day & day of week format in order to be utilized within the ML process
   b. MITRE feature engineering - multi valued MITRE ATT&CK technique string converted into numerical indicators basing on counts of various techniques occuring.
6. One hot encoding - converting all categorical variables into binary numerical vectors 


