import datetime
import random

def get_date_list(last_date, d):
    '''
    :param last_date: the last day we need
    :param d: date interval between the earliest date and last date
    :return: a date list
    '''
    last_date = datetime.datetime.strptime(str(last_date), '%Y%m%d')

    res = []
    while d > 0:
        date_tmp = last_date - datetime.timedelta(days = d)
        res.append(int(date_tmp.strftime('%Y%m%d')))
        d -= 1
    res.append(last_date)

    return res

def train_valid_date_split(date_list, valid_days, num_split):
    '''
    :param date_list: a list of date
    :param valid_days: the number of the days we need in validation set
    :param num_split: how many partition we need in train data set
    :return: validation date set, and a list of train_date set
    '''
    validation_date_list = date_list[-valid_days:]

    random.seed(1024)
    tmp_train = date_list[: -valid_days]
    random.shuffle(tmp_train)

    d = len(tmp_train) // num_split # in this way may lose some days in train dates set
    low = 0
    quick = d
    train_date_list = []
    while quick <= len(tmp_train):
        train_date_list.append(tmp_train[low: quick])
        low = quick
        quick += d

    return validation_date_list, train_date_list

