import pm4py
import numpy as np


def splitRealLog(real, split_size = (0.6, 0.2, 0.2), split_temporal = False, save_to = ''):
    df_real = pm4py.convert_to_dataframe(real)

    cases = df_real['case:concept:name'].unique()
    np.random.seed(72)
    if not split_temporal:
        np.random.shuffle(cases)
    new_case_names = dict(zip(cases,range(len(cases))))
    df_real['CaseN'] = df_real['case:concept:name'].apply(lambda x: new_case_names[x])

    size_train = split_size[0]
    size_val = split_size[1]
    size_test = split_size[2]

    n_train = int(size_train * len(cases))
    n_val = int(size_val * len(cases))
    n_test = int(size_test * len(cases))

    df_train = df_real[df_real['CaseN'] < n_train]
    df_val = df_real[(df_real['CaseN'] >= n_train) & (df_real['CaseN'] < n_train + n_val)]
    df_test = df_real[df_real['CaseN'] >= n_train + n_val]

    del df_train['CaseN']
    del df_val['CaseN']
    del df_test['CaseN']

    real_train = pm4py.convert_to_event_log(df_train) 
    real_val = pm4py.convert_to_event_log(df_val) 
    real_test = pm4py.convert_to_event_log(df_test)

    if save_to:
        pm4py.write_xes(real_train, save_to + '/logTrain.xes')
        pm4py.write_xes(real_val, save_to + '/logVal.xes')
        pm4py.write_xes(real_test, save_to + '/logTest.xes')
    
    return real_train, real_val, real_test