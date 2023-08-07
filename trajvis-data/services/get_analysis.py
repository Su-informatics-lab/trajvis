import matplotlib.pyplot as plt
from math import ceil
import math
import numpy as np
from scipy import linalg, interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return yest
# del 14 15 21 22 23
blue_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863, 
            49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644, 
            59.7568493150685, 64.20958904109591, 61.8061643835616, 57.136301369863, 
            67.9061643835616, 68.77397260273969, 69.6445205479452, 71.1650684931507, 
            70.5671232876712]
blue_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669, 
            95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005, 
            79.66282836588255, 69.0343322902359, 80.7122940246689, 84.29714296400769, 
            53.1459257066338, 49.71462777904635, 42.3471510341773, 45.224953443202, 
            32.78494936494355]

orange_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863, 
            49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644, 
            49.1705479452055, 54.4945205479452, 56.8164383561644, 55.9390410958904, 63.6404109589041, 
            64.9424657534247, 67.9061643835616]
orange_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669, 
            95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005, 
            95.8349884573413, 96.9980674157135, 89.0202724482197, 67.1871752285721, 56.5887043081018, 
            49.273873430824096, 53.1459257066338]

# del 14 15
green_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863, 
            49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644, 
            59.7568493150685, 64.20958904109591, 61.8061643835616, 57.136301369863,  
            67.9061643835616, 64.8356164383562, 69.1157534246575, 72.5178082191781, 
            71.5691780821918, 71.8952054794521, 80.1205479452055, 77.2061643835616, 73.55000000000001, 
            61.7486301369863]
green_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669, 
            95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005, 
            79.66282836588255, 69.0343322902359, 80.7122940246689, 84.29714296400769, 
            53.1459257066338, 75.7299192666146, 67.2039465683612, 73.1266752511189, 
            70.7468372579639, 57.609710695723706, 53.1720428523801, 45.1267891217517, 52.42981241600655, 
            70.4927317098041]

def getFittingPoints(age, egfr):
    n = 100
    x_y = list(map(lambda x, y: [x, y], age, egfr))
    x_y.sort()
    x_y = np.array(x_y)
   
    x = x_y[:, 0]
    y = x_y[:, 1]
    
    f = 0.5
    yest = lowess(x, y, f=f, iter=6)
    
    res = list(map(lambda x, y: [round(x,2), round(y, 2)], x, yest))
    return res

def getTrajectoryPoints():
    return {
        'blue':getFittingPoints(blue_age, blue_egfr), 
        'green':getFittingPoints(green_age, green_egfr),
        'orange':getFittingPoints(orange_age, orange_egfr)
    }
features_all_csn_df = pd.read_csv('data/features_all_csn_id.csv', delimiter=",")
embeddings_all_id_df = pd.read_csv('data/embeddings_all_id_cluster.csv', delimiter=",")
ordered_feats = pd.read_csv('data/ordered_feats.csv', delimiter=",")
outputval_try = np.load('data/graphsage_output.npy')
outputval_try = pd.DataFrame(outputval_try)

greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

train_set = set(blueline)
train_set.update(orangeline)
train_set.update(greenline)

from sklearn.neighbors import KNeighborsClassifier
neigh_graphsage = KNeighborsClassifier(n_neighbors=5)
neigh_graphsage.fit(pd.DataFrame(outputval_try), np.ravel(ordered_feats['cluster_label'].values))

def fakeBlueData(inputx_s, widths):
    age, egfr = blue_age, blue_egfr
    data = np.round(np.array(getFittingPoints(age, egfr)),decimals=2)
    tck = interpolate.splrep(data[:, 0], data[:, 1])
    return getTwoPosition(inputx_s, widths, tck)

def fakegreenData(inputx_s, widths):
    age, egfr = green_age, green_egfr
    data = np.round(np.array(getFittingPoints(age, egfr)),decimals=2)
    tck = interpolate.splrep(data[:, 0], data[:, 1])
    return getTwoPosition(inputx_s, widths, tck)

def fakeOrangeData(inputx_s, width):
    age, egfr = orange_age, orange_egfr
    data = np.round(np.array(getFittingPoints(age, egfr)),decimals=2)
    tck = interpolate.splrep(data[:, 0], data[:, 1])
    return getTwoPosition(inputx_s, width, tck)

def get_one_pat_cluster_label_by_age(id):
    features_id = features_all_csn_df[features_all_csn_df['pat_id']==id]
    id_csn = list(features_id.csn)
    embedding_id = embeddings_all_id_df[embeddings_all_id_df['csn'].isin(id_csn)]
    ages = []
    for i in embedding_id.csn:
        age = list(features_id[features_id.csn==i].age)[0]
        ages.append(age)
    embedding_id.insert(1, 'age', ages)
    embedding_id = embedding_id.sort_values('age')
    
    ppt_cluster_label = []
    for csn_num in embedding_id.csn:
        if csn_num in list(ordered_feats.csn):
            ppt_cluster_label.append(list(ordered_feats[ordered_feats.csn==csn_num].cluster_label)[0])
        else:
            temp = neigh_graphsage.predict(embedding_id[embedding_id.csn == csn_num].iloc[:,6:])[0]
            ppt_cluster_label.append(temp)
    green_label = []
    blue_label = []
    orange_label = []
    for label in ppt_cluster_label:
        if label in blueline:
            blue_label.append(1)
        else:
            blue_label.append(0)
        if label in greenline:
            green_label.append(1)
        else:
            green_label.append(0)
        if label in orangeline:
            orange_label.append(1)
        else:
            orange_label.append(0)
    d = {'age': embedding_id['age'], 'csn': embedding_id['csn'], 'cluster_label': ppt_cluster_label, 'green_label':green_label, 'blue_label':blue_label, 'orange_label':orange_label}
    pat_cluster = pd.DataFrame(data=d)
    
    ages = [int(list(pat_cluster.age)[0])]
#     green_width = [15]
#     blue_width = [15]
#     orange_width = [15]
    green_width = [33]
    blue_width = [33]
    orange_width = [33]
    
    for idx, row in pat_cluster.iterrows():
        label=row['cluster_label']
        if label in train_set:
            ages.append(round(row['age'], 2))
            num = row['green_label']+row['blue_label']+row['orange_label']
            res = 3-num
            if row['green_label']>0:
                # green_width.append(green_width[-1]-1/num)
                green_width.append(round(green_width[-1]+1/num, 2))
            else:
                if green_width[-1] < 0:
                    green_width.append(0)
                else: 
                    green_width.append(round(green_width[-1]-1/res, 2))
                # green_width.append(green_width[-1]+1/res)
            if row['blue_label']>0:
                blue_width.append(round(blue_width[-1]+1/num,2))
                # blue_width.append(blue_width[-1]-1/num)
            else:
                if blue_width[-1] < 0:
                    blue_width.append(0)
                else: 
                    blue_width.append(round(blue_width[-1]-1/res,2))
                # blue_width.append(blue_width[-1]+1/res)
            if row['orange_label']>0:
                orange_width.append(round(orange_width[-1]+1/num,2))
                # orange_width.append(orange_width[-1]-1/num)
            else:
                if orange_width[-1] < 0:
                    orange_width.append(0)
                else: 
                    orange_width.append(round(orange_width[-1]-1/res,2))
                # orange_width.append(orange_width[-1]+1/res)
    j = ages[-1]
    age_last = ages[-1]
    plus = 0
    minors = 0
    orange_last = orange_width[-1]
    blue_last = blue_width[-1]
    green_last = green_width[-1]
    while j < 80:
        j = j + 1
        plus+=0.5
        minors+=1/4
        ages.append(j)
        if orange_last+plus>100 or green_last+plus>100 or blue_last+plus>100:
            orange_width.append(orange_last)
            blue_width.append(blue_last)
            green_width.append(green_last)
            continue
        if orange_last >= blue_last and orange_last >= green_last:
            if green_width[-1]<0:
                green_width.append(0)
            else:
                green_width.append(green_last-minors)
            if blue_width[-1]<0:
                blue_width.append(0)
            else:
                blue_width.append(blue_last-minors)
            orange_width.append(orange_last+plus)
        elif blue_last >= orange_last and blue_last >= green_last:
            if green_width[-1]<0:
                green_width.append(0)
            else:
                green_width.append(green_last-minors)
            blue_width.append(blue_last+plus)
            if orange_width[-1]<0:
                orange_width.append(0)
            else:
                orange_width.append(orange_last-minors)
        elif green_last >= orange_last and green_last >= blue_last:
            green_width.append(green_last+plus)
            if blue_width[-1]<0:
                blue_width.append(0)
            else:
                blue_width.append(blue_last-minors)
            if orange_width[-1]<0:
                orange_width.append(0)
            else:
                orange_width.append(orange_last-minors)            
    return age_last, ages, green_width, blue_width, orange_width

def get_three_uncertainty(id):
    features_id = features_all_csn_df[features_all_csn_df['pat_id']==id]
    id_csn = list(features_id.csn)
    embedding_id = embeddings_all_id_df[embeddings_all_id_df['csn'].isin(id_csn)]
    ages = []
    for i in embedding_id.csn:
        age = list(features_id[features_id.csn==i].age)[0]
        ages.append(age)
    embedding_id.insert(1, 'age', ages)
    embedding_id = embedding_id.sort_values('age')
    
    ppt_cluster_label = []
    for csn_num in embedding_id.csn:
        if csn_num in list(ordered_feats.csn):
            ppt_cluster_label.append(list(ordered_feats[ordered_feats.csn==csn_num].cluster_label)[0])
        else:
            temp = neigh_graphsage.predict(embedding_id[embedding_id.csn == csn_num].iloc[:,6:])[0]
            ppt_cluster_label.append(temp)
    green_label = []
    blue_label = []
    orange_label = []
    for label in ppt_cluster_label:
        if label in blueline:
            blue_label.append(1)
        else:
            blue_label.append(0)
        if label in greenline:
            green_label.append(1)
        else:
            green_label.append(0)
        if label in orangeline:
            orange_label.append(1)
        else:
            orange_label.append(0)
    d = {'age': embedding_id['age'], 'csn': embedding_id['csn'], 'cluster_label': ppt_cluster_label, 'green_label':green_label, 'blue_label':blue_label, 'orange_label':orange_label}
    pat_cluster = pd.DataFrame(data=d)
    
    ages = [int(list(pat_cluster.age)[0])]
    green_width = [30]
    blue_width = [30]
    orange_width = [30]
    min_width = 10
    max_width = 50
    for idx, row in pat_cluster.iterrows():
        label=row['cluster_label']
        if label in train_set:
            ages.append(round(row['age'], 2))
            num = row['green_label']+row['blue_label']+row['orange_label']
            res = 3-num
            if row['green_label']>0:
                if green_width[-1] < min_width:
                    green_width.append(min_width)
                else: 
                    green_width.append(round(green_width[-1]-1/num, 2))
            else:
                green_width.append(round(green_width[-1]+1/num, 2))
            if row['blue_label']>0:
                if blue_width[-1] < min_width:
                    blue_width.append(min_width)
                else: 
                    blue_width.append(round(blue_width[-1]-1/num,2))
            else:
                blue_width.append(round(blue_width[-1]+1/num,2))
            if row['orange_label']>0:
                if orange_width[-1] < min_width:
                    orange_width.append(min_width)
                else: 
                    orange_width.append(round(orange_width[-1]-1/num,2))
            else:
                orange_width.append(round(orange_width[-1]+1/num,2))
    j = ages[-1]
    age_last = ages[-1]
    plus = 0
    minors = 0
    orange_last = orange_width[-1]
    blue_last = blue_width[-1]
    green_last = green_width[-1]
    while j < 80:
        j = j + 1
        plus+=0.5
        minors+=1/4
        ages.append(j)
        if orange_last-plus<min_width or green_last-plus<min_width or blue_last-plus<min_width:
            green_width.append(green_last)
            orange_width.append(orange_last)
            blue_width.append(blue_last)
            continue
        if orange_last < blue_last and orange_last < green_last:
            if green_width[-1]<max_width:
                green_width.append(green_last+minors)
            else:
                green_width.append(max_width)
            if blue_width[-1]<max_width:
                blue_width.append(blue_last+minors)
            else:
                blue_width.append(max_width)
            if orange_last < 10:
                orange_width.append(10)
            else:
                orange_width.append(orange_last-plus)
        elif blue_last < orange_last and blue_last < green_last:
            if green_width[-1]<max_width:
                green_width.append(green_last+minors)
            else:
                green_width.append(max_width)
            if orange_width[-1]<max_width:
                orange_width.append(orange_last+minors)
            else:
                blue_width.append(max_width)
            if blue_width[-1] < 10:
                blue_width.append(10)
            else:
                blue_width.append(blue_last-plus)
        elif green_last < blue_last and green_last < orange_last:
            if orange_width[-1]<max_width:
                orange_width.append(orange_last+minors)
            else:
                orange_width.append(max_width)
            if blue_width[-1]<max_width:
                blue_width.append(blue_last+minors)
            else:
                blue_width.append(max_width)
            if green_width[-1] < 10:
                green_width.append(10)
            else:
                green_width.append(green_last-plus)   
    green_width = list(np.clip(green_width, min_width, max_width))
    blue_width = list(np.clip(blue_width, min_width, max_width))
    orange_width = list(np.clip(orange_width, min_width, max_width))
    return ages, green_width, blue_width, orange_width


def getThreePossibility(ages, green_width, blue_width, orange_width):
    merge_df = pd.DataFrame({'age':ages, 'green_width': green_width, 'blue_width': blue_width, 'orange_width':orange_width})
    merge_df.age = merge_df.age.astype(int)

    agg_functions = {'green_width': 'mean', 'blue_width': 'mean', 'orange_width': 'mean'}
    df_new = merge_df.groupby(merge_df['age'], as_index=False).aggregate(agg_functions)

    for age in range(30,df_new.age.max()):
        if age < df_new.age.min():
            dict_tmp = {'age': age, 'green_width': 0.33, 'blue_width': 0.33, 'orange_width': 0.33}
            df_new = df_new._append(dict_tmp, ignore_index=True)
        elif age not in list(df_new.age):
            dict_tmp = {'age': age, 'green_width': np.NaN, 'blue_width': np.NaN, 'orange_width': np.NaN}
            df_new = df_new._append(dict_tmp, ignore_index=True)
    df_new = df_new.sort_values(by='age', ascending=True)
    df_new =df_new.interpolate()
    df_new.iloc[:,1:] = df_new.iloc[:,1:].div(df_new.iloc[:,1:].sum(axis=1), axis=0)
    df_new = df_new.round(2)
    return list(df_new.age), list(df_new.green_width), list(df_new.blue_width), list(df_new.orange_width)

def getTwoPosition(input_xs, widths, tck):
    # res_up = []
    # res_dn = []
    res = []
    for index in range(len(input_xs)):
        input_x = input_xs[index]
        width = widths[index]
        x0 = input_x-0.1
        y0 = pred_val(x0, tck)
        x1 = input_x+0.1
        y1 = pred_val(x1, tck)
        input_y = pred_val(input_x, tck)
        slope = (y1-y0)/(x1-x0)
        # theta = math.atan(-1/slope)
        theta = math.pi/2
        output_x1 = input_x + math.cos(theta)*width/2
        output_y1 = input_y + math.sin(theta)*width/2
        output_x2 = input_x - math.cos(theta)*width/2
        output_y2 = input_y - math.sin(theta)*width/2
        # res_up.append([output_x1, output_y1])
        # res_dn.append([output_x2, output_y2])
        res.append([(output_x1+output_x2)/2, output_y1, output_y2])
    return res

def pred_val(x, tck):
    return interpolate.splev(x, tck)

pat_traj = pd.read_csv('data/pat_traj.csv', delimiter=",")
ckd_crf_demo = pd.read_csv('data/ckd_crf_demo.csv', delimiter=",")


def get_pat_sex_distribution():
    res = []
    for traj in ['orange', 'blue', 'green']:
        if traj=='all':
            pats = pat_traj.pat_id
        else:
            pats = pat_traj[pat_traj['traj']==traj].pat_id
        pat_demo = ckd_crf_demo[ckd_crf_demo['pat_id'].isin(pats)]
        sex_value_counts = pat_demo.sex_cd.value_counts()
        F_num = sex_value_counts.F
        M_num = sex_value_counts.M
        res.append([traj, round(F_num/(F_num+M_num), 2), round(M_num/(F_num+M_num), 2)])
    return res

def get_pat_race_distribution():
    res = []
    for traj in ['orange', 'blue', 'green']:
        if traj=='all':
            pats = pat_traj.pat_id
        else:
            pats = pat_traj[pat_traj['traj']==traj].pat_id
        pat_demo = ckd_crf_demo[ckd_crf_demo['pat_id'].isin(pats)]
        race_value_counts = pat_demo.race_cd.value_counts()
        B_num = race_value_counts.B
        W_num = race_value_counts.W
        res.append([traj, round(B_num/(B_num+W_num),2), round(W_num/(B_num+W_num),2)])
    return res

ckd_data_df=pd.read_csv('data/ckd_emr_data.csv',skipinitialspace = True, delimiter=",")

def get_concept_distribution(concept):
    x_value = []
    y_value = []
    for traj in ['orange', 'blue', 'green']:
        pats = pat_traj[pat_traj['traj']==traj].pat_id
        records = ckd_data_df[ckd_data_df['pat.id'].isin(pats)]
        target = records[records['concept.cd']==concept]   
        x_s = []
        res = []
        i = 30
        while i < 90:
            x_s.append(i)
            value = target[(target['age']>i) & (target['age']<i+5)]['nval.num'].mean()
            value = round(value, 2)
            i += 5
            if pd.isna(value):
                value = 0
            res.append(round(value, 2))
        y_value.append([traj,res])
        x_value = x_s
    return x_value, y_value