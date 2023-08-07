import pandas as pd

embeddings_all_csn = pd.read_csv('data/embeddings_all_id_cluster.csv',skipinitialspace = True )
features_all_csn = pd.read_csv('data/features_all_csn_id.csv',skipinitialspace = True)

greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

after_green = [60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
after_blue = [0, 99, 58, 29, 47, 82, 67, 14]

after_orange = [49, 85, 72, 34, 25, 10, 33]
after_green_blue = [89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 0, 99]

before_1 = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16]
before_2 = [89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33]

green_points = features_all_csn[features_all_csn['cluster_label'].isin(greenline)]
orange_points = features_all_csn[features_all_csn['cluster_label'].isin(orangeline)]
blue_points = features_all_csn[features_all_csn['cluster_label'].isin(blueline)]

after_green_points = features_all_csn[features_all_csn['cluster_label'].isin(after_green)]
after_blue_points = features_all_csn[features_all_csn['cluster_label'].isin(after_blue)]

after_orange_points = features_all_csn[features_all_csn['cluster_label'].isin(after_orange)]
after_green_blue_points = features_all_csn[features_all_csn['cluster_label'].isin(after_green_blue)]

before_1_points = features_all_csn[features_all_csn['cluster_label'].isin(before_1)]
before_2_points = features_all_csn[features_all_csn['cluster_label'].isin(before_2)]

# The first common line
to_orange_pat = list(set(before_1_points.pat_id).intersection(set(after_orange_points.pat_id)))
to_green_blue_pat = list(set(before_1_points.pat_id).intersection(set(after_green_blue_points.pat_id)))
to_orange_pat_points = before_1_points[before_1_points['pat_id'].isin(to_orange_pat)]
to_green_blue_pat_points = before_1_points[before_1_points['pat_id'].isin(to_green_blue_pat)]

# The second common line: Find the patient go to green and blue line
to_green_pat = list(set(before_2_points.pat_id).intersection(set(after_green_points.pat_id)))
to_blue_pat = list(set(before_2_points.pat_id).intersection(set(after_blue_points.pat_id)))
to_green_pat_points = before_2_points[before_2_points['pat_id'].isin(to_green_pat)]
to_blue_pat_points = before_2_points[before_2_points['pat_id'].isin(to_blue_pat)]

before_1_class = []
for csn_num in before_1_points.index:
    if csn_num in to_orange_pat_points.index:
        before_1_class.append("orange")
    elif csn_num in to_green_blue_pat_points.index:
        before_1_class.append("green_blue")
    else:
        before_1_class.append("TBD")

if('before_1_class' in before_1_points.columns):
    del before_1_points["before_1_class"]
before_1_points.insert(0, 'before_1_class', before_1_class)

before_2_class = []
for csn_num in before_2_points.index:
    if csn_num in to_green_pat_points.index:
        before_2_class.append("orange")
    elif csn_num in to_blue_pat_points.index:
        before_2_class.append("blue")
    else:
        before_2_class.append("TBD")

if('before_2_class' in before_1_points.columns):
    del before_1_points["before_2_class"]
before_2_points.insert(0, 'before_2_class', before_2_class)

raw_embedding = embeddings_all_csn.iloc[:, 5: ]

training_set_1 = before_1_points[before_1_points['before_1_class'].isin(['green_blue', 'orange'])]
data_training_set_1 = raw_embedding[raw_embedding.index.isin(training_set_1.index)]
test_set_1 = before_1_points[before_1_points['before_1_class'].isin(['TBD'])]
data_test_set_1 = raw_embedding[raw_embedding.index.isin(test_set_1.index)]

training_set_2 = before_2_points[before_2_points['before_2_class'].isin(['green', 'blue'])]
data_training_set_2 = raw_embedding[raw_embedding.index.isin(training_set_2.index)]
test_set_2 = before_2_points[before_2_points['before_2_class'].isin(['TBD'])]
data_test_set_2 = raw_embedding[raw_embedding.index.isin(test_set_2.index)]

from sklearn.neighbors import KNeighborsClassifier
neigh1 = KNeighborsClassifier(n_neighbors=5)
neigh1.fit(pd.DataFrame(data_training_set_1), np.ravel(training_set_1['before_1_class'].values))

neigh2 = KNeighborsClassifier(n_neighbors=5)
neigh2.fit(pd.DataFrame(data_training_set_2), np.ravel(training_set_2['before_2_class'].values))

new_class_list = []
for csn_num in before_1_points.index:
    if csn_num in to_orange_pat_points.index:
        new_class_list.append("orange")
    elif csn_num in to_green_blue_pat_points.index:
        new_class_list.append("green_blue")
    else:
        one_row = before_1_points[before_1_points.index == csn_num]
        new_class_list.append(neigh1.predict(raw_embedding[raw_embedding.index == list(one_row.index)[0]])[0])

if('new_class_list' in before_1_points.columns):
    del before_1_points["new_class_list"]
before_1_points.insert(0, 'new_class_list', new_class_list)

new_class_list = []
for csn_num in before_2_points.index:
    if csn_num in to_green_pat_points.index:
        new_class_list.append("green")
    elif csn_num in to_blue_pat_points.index:
        new_class_list.append("blue")
    else:
        one_row = before_2_points[before_2_points.index == csn_num]
        new_class_list.append(neigh2.predict(raw_embedding[raw_embedding.index == list(one_row.index)[0]])[0])

if('new_class_list' in before_2_points.columns):
    del before_1_points["new_class_list"]
before_2_points.insert(0, 'new_class_list', new_class_list)

import numpy as np
from scipy.stats import chi2_contingency

def ava_p_val(df_1, df_2):
    column_name = ['var_name', 'df_1_0', 'df_1_1', 'df_2_0', 'df_2_1', 'chi2', 'dof', 'p_val']
    res_df = pd.DataFrame(columns=column_name)
    for i in range(len(var_ava)):
        var = var_ava[i]
        x1 = df_1[df_1[var]==0]
        x1 = len(x1)   
        y1 = df_1[df_1[var]==1]
        y1 = len(y1)
        x2 = df_2[df_2[var]==0]
        x2 = len(x2)
        y2 = df_2[df_2[var]==1]
        y2 = len(y2)
        if y1==0 or y2==0:
            print(var, x1, y1, x2, y2)
        else:
            obs = np.array([[x1, y1], [x2, y2]])
            chi2, p, dof, ex = chi2_contingency(obs)
            res_df = res_df.append({'var_name':var, 'df_1_0':x1, 'df_1_1':y1, 'df_2_0':x2, 'df_2_1':y2, 'chi2':chi2, 'dof':dof, 'p_val': p}, ignore_index=True)
    display(res_df)
    fdr = getfdr(res_df.p_val)

var_ava = ['ALK_avail', 'ALT_SGPT_avail',
           'AST_SGOT_avail','BP_DIASTOLIC_avail',
           'BP_SYSTOLIC_avail', 'CHOLESTEROL_avail', 
           'CREATINE_KINASE_avail','EGFR_avail', 
           'HBA1C_avail', 'HDL_avail',
           'HEMOGLOBIN_avail', 'HT_avail', 'INR_avail',
           'LDL_avail','TBIL_avail',
           'TRIGLYCERIDES_avail', 'TROPONIN_avail', 
           'WT_avail']
var_val = [
            'ALK_val', 'ALT_SGPT_val', 
            'AST_SGOT_val', 'BP_DIASTOLIC_val', 
            'BP_SYSTOLIC_val', 'CHOLESTEROL_val', 
            'CREATINE_KINASE_val', 'EGFR_val', 
            'HBA1C_val', 'HDL_val',
            'HEMOGLOBIN_val','HT_val', 'INR_val', 
            'LDL_val', 'TBIL_val', 
            'TRIGLYCERIDES_val', 'TROPONIN_val', 
            'WT_val', 
          ]

import scipy.stats as stats
def val_p_val(df_1, df_2, color_1, color_2, group):
    p_vals = []
    stat = []
    col_name = ['group', 'var_name', 'len_x', 'len_y', 'mean_x', 'mean_y', 'stat', 'p_val']
    res_df = pd.DataFrame(columns=col_name)
    for i in range(len(var_ava)):
        avail = var_ava[i]
        var = var_val[i]
        x = np.array(df_1[df_1[avail]>=0.00001][var])
        y = np.array(df_2[df_2[avail]>=0.00001][var])
        if len(x)<=1 or len(y)<=1:
#             res_df = res_df.append({'group': group, 'var_name':var, 'len_x':len(x), 'len_y':len(y), 
#                                     'mean_x':x.mean(), 'mean_y':y.mean(),
#                                     'stat':'NA', 'p_val': 'NA'}, ignore_index=True)
            p_vals.append('NA')
            stat.append('NA')
        else:
            res = stats.ttest_ind(a=x, b=y)
            p_vals.append(res.pvalue)
            stat.append(res.statistic)
            res_df = res_df.append({'group': group, 'color_1': color_1, 'color_2':color_2, 
                                    'var_name':var, 'len_x':len(x), 'len_y':len(y), 
                                    'mean_x':x.mean(), 'mean_y':y.mean(),
                                    'stat':res.statistic, 'p_val': res.pvalue}, ignore_index=True)
#     fdr = getfdr(res_df.p_val)
    return res_df
#     plotVocano(stat, fdr[1])

#Significance test for before green_blue and orange
green_blue_df = before_1_points[before_1_points['new_class_list'] == 'green_blue']
orange_df = before_1_points[before_1_points['new_class_list'] == 'orange']
# ava_p_val(green_blue_df, orange_df)
before_1_p = val_p_val(green_blue_df, orange_df, 'green,blue', 'orange', 'before_1')

# Significant test for before green and blue
green_before_df = before_2_points[before_2_points['new_class_list'] == 'green']
blue_before_df = before_2_points[before_2_points['new_class_list'] == 'blue']

# ava_p_val(green_before_df, blue_before_df)
before_2_p = val_p_val(green_before_df, blue_before_df, 'green', 'blue',  'before_2')

# Significant test for green and blue
# ava_p_val(after_green_points, after_blue_points)
after_green_after_blue_p = val_p_val(after_green_points, after_blue_points,'green', 'blue',  'after_green_after_blue')
# ava_p_val(after_orange_points, after_green_blue_points)
after_orange_after_green_blue_p = val_p_val(after_orange_points, after_green_blue_points, 
                                            'orange', 'green,blue',
                                            'after_orange_after_green_blue')

look_up_p = pd.concat([before_1_p, before_2_p, after_green_after_blue_p, after_orange_after_green_blue_p], axis=0)
look_up_p