from tqdm.notebook import trange, tqdm
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from tqdm.notebook import trange, tqdm
import json
import pandas as pd
import numpy as np
import sys
import public_function as pf


class LogEvent:
    # def __init__(self, config, method):
    #     self.config = config
    #     self.method = method
    #     self.methods_list = {'log_scale': self.log_scale,
    #                          'random_sampling': self.random_sampling,
    #                          'stratified_sampling': self.stratified_sampling}
    #     assert self.method in self.methods_list.keys()
    def __init__(self):
        print('begin')

    @staticmethod
    def log_scale(df, run_table, log_files, save_path):
        logs_list = []
        for i in range(0, len(run_table)):
            service_list = []
            for j in range(len(df)):
                duration = int(run_table['duration'][i] / 60)
                if duration == 0:
                    duration = 1
                for k in range(duration):
                    st_time = run_table['st_time'][i]
                    st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S.%f")
                    st_stamp = int(time.mktime(st_array.timetuple()) * 1000.0 + st_array.microsecond / 1000.0)
                    new_stamp = st_stamp + k * 60000 + 30000

                    temp_df = df[j].loc[(df[j]['timestamp'] >= (st_stamp + k * 60000)) & (
                            df[j]['timestamp'] < (st_stamp + (k + 1) * 60000))]
                    unique_list = np.unique(temp_df['EventId'], return_counts=True)
                    event_id = unique_list[0]
                    cnt = unique_list[1]

                    for t in range(len(event_id)):
                        str_cnt = str(cnt[t])
                        event = event_id[t] + '_' + str_cnt
                        service_list.append([new_stamp, log_files[j], event])
            logs_list.append(service_list)
        np.save(save_path, logs_list)

    @staticmethod
    def random_sampling(df, run_table, save_path):
        logs_list = []
        for i in range(0, len(run_table)):
            service_list = []
            for j in range(len(df)):
                temp_df = df[j].loc[(df[j]['datetime'] >= run_table['st_time'][i])
                                    &
                                    (df[j]['datetime'] <= run_table['ed_time'][i])]
                if len(temp_df) == 0:
                    continue
                elif 11 > len(temp_df) > 0:
                    temp_df = temp_df
                elif 1001 > len(temp_df) >= 11:
                    temp_df = temp_df.sample(n=10)
                elif 2000 > len(temp_df) >= 1001:
                    tmp = int(len(temp_df) / 100)
                    temp_df = temp_df.sample(n=tmp)
                else:
                    temp_df = temp_df.sample(n=20)

                print(len(temp_df))
                for _, row in temp_df.iterrows():
                    st_time = row['datetime']
                    st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S.%f")
                    st_stamp = int(time.mktime(st_array.timetuple()) * 1000.0 + st_array.microsecond / 1000.0)
                    service_list.append([st_stamp, row['Service'], row['EventId']])
            logs_list.append(service_list)
        np.save(save_path, logs_list)

    @staticmethod
    def stratified_sampling(df, run_table, log_files, save_path):
        logs_list = []
        # for i in tqdm(range(len(labels))):
        for i in range(0, len(run_table)):
            # for i in range(250,475):
            service_list = []
            for j in range(len(df)):
                temp_df = df[j].loc[(df[j]['datetime'] >= run_table['st_time'][i])
                                    &
                                    (df[j]['datetime'] <= run_table['ed_time'][i])]
                unique_list = np.unique(temp_df['EventId'], return_counts=True)
                event_id = unique_list[0]
                cnt = unique_list[1]
                for k in range(len(cnt)):
                    if cnt[k] == 1:
                        unique_time = temp_df['datetime'].loc[temp_df['EventId'] == event_id[k]]
                        unique_array = datetime.strptime((list(unique_time))[0], "%Y-%m-%d %H:%M:%S.%f")
                        unique_stamp = int(
                            time.mktime(unique_array.timetuple()) * 1000.0 + unique_array.microsecond / 1000.0)
                        service_list.append([unique_stamp, log_files[j], event_id[k]])
                        temp_df = temp_df.loc[temp_df['EventId'] != event_id[k]]
                X = temp_df
                y = temp_df['EventId']
                if len(temp_df) == 0:
                    continue
                elif 21 > len(temp_df) >= 1:
                    X_test = temp_df
                else:
                    tmp = len(event_id)
                    if tmp < 20:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tmp, stratify=y)
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, stratify=y)
                # print(X_test)
                for _, row in X_test.iterrows():
                    st_time = row['datetime']
                    st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S.%f")
                    st_stamp = int(time.mktime(st_array.timetuple()) * 1000.0 + st_array.microsecond / 1000.0)
                    service_list.append([st_stamp, row['Service'], row['EventId']])
            logs_list.append(service_list)
        np.save(save_path, logs_list)

    def do_lab(self, path):
        log_files = ['cartservice', 'checkoutservice', 'currencyservice', 'emailservice',
                     'frontend', 'paymentservice', 'productcatalogservice', 'recommendationservice',
                     'shippingservice']
        df = np.load(path, allow_pickle=True)
        df = list(df)
        run_table = pd.read_csv('./data/run_table.csv')
        self.log_scale(df, run_table, log_files, 'log_scale.npy')
        print('log_scale_finish')
        self.random_sampling(df, run_table, 'random.npy')
        print('random finish')
        self.stratified_sampling(df, run_table, log_files, 'stratification.npy')
        print('stratified finish')