from dataLoader import getData, getDateList
from id_2_index import id_to_index
from multi_task import CGC_Model, layerEmbedding
import os
import logging
import sys
import subprocess
from pyhive import hive
import pandas as pd
import time
import datetime

import math
import warnings
import json
import numpy as np
from time import time, localtime
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import *
from sklearn.metrics import roc_auc_score
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import save_model, load_model
import tensorflow.keras as keras

# !kinit -kt /root/user.keytab USER
item_id_sql = '' # sql used to select item id from table
origin_order_field = getData.get_data_from_hive(item_id_sql)

item_id_full = list(set())
itemid_to_id = dict(
    [('<pad>', 0)] + [(item, id_to_index.get_hash_number(item)) for item in item_id_full]
)

# save the dict as csv
pd.DataFrame.from_dict(itemid_to_id, orient='index').to_csv('itemid_2_index.csv')

# get the past 60days dates
past_day_num = 60
now = datetime.datetime.now().strftime('%Y%m%d')
date_list = getDateList.get_date_list(now, past_day_num)

# split the latest m dates as validation dates set and the other dates as num_split group
num_split = 2
valid_days = 18
tuning_valid_date, train_date = getDateList.train_valid_date_split(date_list, valid_days, num_split)

# split tuning_valid_date as fine-tuning train set and valid set
tuning_date_num = 14
tuning_date, valid_date = tuning_valid_date[:tuning_date_num], tuning_valid_date[tuning_date_num:]

# get tuning and validation data
valid_sql = ('.. %s' %(str(tuple(valid_date))))
valid_data_ori = getData.get_data_from_hive(valid_sql, col_name_format=False)

# Data Processing

# columns list of different types of data
cate_col = [] #Discrete data
binary_col = [] #binary data
num_col = [] #continuous data

# Handle shopping cart item strings
def _bulid_trunc(s, num, remap):
    '''
    a function used to transfer the item id to a numerical id
    :param s: list of item id list(str())
    :param num: number of the item id we need
    :param remap: True or False, if True the item id will be mapped to a numerical id else will not
    :return: a list of id(int)
    '''
    res = [0] * num
    s = s.split(',')
    random.shuffle(s)

    for i in range(min(len(s), num)):
        if remap == True:
            res[i] = itemid_to_id[s[i]]
        if remap == False:
            res[i] = s[i]

    return  res

def trunc_order(orders, num, remap):
    '''
    transfer item ids in the order list to id
    :param orders: list of item ids
    :param num: same as _bulid_trunc
    :param remap: same as _bulid_trunc
    :return: a dataframe of id with column name like 'order_1'...
    '''
    cols = ['order_' + str(i) for i in range(1, num+1)]
    cand = []

    orderList = list(orders)
    random.shuffle(orderList)

    for s in orderList:
        if remap == True:
            cand.append(_bulid_trunc(s, num, True))
        if remap == False:
            cand.append(_bulid_trunc(s, num, False))
    cand = np.stack(cand)

    return pd.DataFrame(cand, columns=cols)

# data process for validation data
label_sql = ''
merged_label = getData.get_data_from_hive(label_sql %(str(tuple(valid_date))))
undersample = random.sample(list(merged_label[merged_label['hit'] == 0].index),
                           int(len(merged_label[merged_label['hit'] == 0]) * 0.25)) + list(merged_label[merged_label['hit'] == 1].index) #the size of negative set is 25% of negative data and 100% of positive data
random.shuffle(undersample)
df_undersample = merged_label.loc[undersample]


