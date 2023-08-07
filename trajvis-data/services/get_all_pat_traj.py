import pandas as pd
import numpy as np

greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

features_all_csn = pd.read_csv('data/features_all_csn_id.csv',skipinitialspace = True)

col_name = ['pat_id', 'traj']
res_df = pd.DataFrame(columns=col_name)
counter = 0
for pat_id in features_all_csn.pat_id:
    counter = counter+1
    if counter % 1000 == 0:
        print(counter)
    individual_df = features_all_csn[features_all_csn['pat_id']==pat_id]
    orange_num = 0
    blue_num = 0
    green_num = 0
    traj = None
    for label in individual_df.cluster_label:
        if label in orangeline:
            orange_num = orange_num+1
        if label in blueline:
            blue_num = blue_num+1
        if label in greenline:
            green_num = green_num+1
    if orange_num > blue_num and orange_num > green_num:
        traj = 'orange'
    elif blue_num > orange_num and blue_num > green_num:
        traj = 'blue'
    else:
        traj = 'green'
    res_df = res_df.append({'pat_id': pat_id, 'traj':traj}, ignore_index=True)

res_df.to_csv('pat_traj.csv')

