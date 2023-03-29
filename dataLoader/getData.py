# read data from hive or hdfs
import pandas as pd
from pyhive import hive

# config
TaskId = ''
HADOOP_HOST = 'host.com'
HADOOP_QUEUE = 'root'
HADOOP_USER_NAME = 'myself'
hive_database = 'mtl_db'
remote_path = '/user/myself/'

def get_data_from_hive(sql_str, col_name_format = True):
    '''
    :param sql_str: a sql string to select data from hive table
    :param col_name_format: the columns of the table
    :return: a pandas dataframe
    '''
    hive_config = {
        'mapreduce.job.queuename': HADOOP_QUEUE,
        'hive.exec.compress.output': 'false'
    }

    connection = hive.connect(host = HADOOP_HOST,
                              port = 10000,
                              username = HADOOP_USER_NAME,
                              database = hive_database,
                              auth = 'KERBEROS',
                              keberos_service_name = 'hive',
                              configuration = hive_config)

    data = pd.read_sql(sql_str, connection)

    if col_name_format:
        data.columns = list(map(lambda x: x.split('.')[1], data_columns))
    else:
        pass

    return data