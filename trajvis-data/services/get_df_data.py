import pandas as pd
import numpy as np
import datetime
import random
from dateutil.relativedelta import relativedelta
ckd_data_df= pd.read_csv('data/ckd_emr_data.csv',skipinitialspace = True, delimiter="," )
ckd_crf_demo = pd.read_csv('data/ckd_crf_demo.csv', delimiter=",")
ordered_feats = pd.read_csv('data/ordered_feats.csv', delimiter=",")
features_all_csn = pd.read_csv('data/features_all_csn_id.csv',delimiter=",", skipinitialspace = True)

def get_pat_records(id):
    return ckd_data_df[ckd_data_df['pat.id']==id]

def get_profile_date(pat_id):
    ages = get_pat_records(pat_id).age
    cur_age = int(max(ages))
    in_date = '2015-6-15'
    dt = datetime.datetime.strptime(in_date, "%Y-%m-%d")
    n = random.randrange(360)   
    last_visit_res = (dt + datetime.timedelta(days=n)).strftime("%Y-%m-%d")
    last_visit_date = datetime.datetime.strptime(last_visit_res,"%Y-%m-%d")
    rand_days = random.randrange(180)   

    birth_day = (last_visit_date - relativedelta(years=cur_age)).strftime("%Y-%m-%d")
    birth_day = datetime.datetime.strptime(birth_day,"%Y-%m-%d")
    birth_day = (birth_day - datetime.timedelta(days=rand_days)).strftime("%Y-%m-%d")

    return [birth_day, last_visit_res]

def get_pat_unique_concept(id):
    df = get_pat_records(id)
    df = df.sort_values(by=['age'])
    concepts = (df['concept.cd'].unique())
    res_list = []
    for i in concepts:
        ages = list(df[df['concept.cd']==i]['age'])
        vals = list(df[df['concept.cd']==i]['nval.num'])
        age_val = list(map(lambda x,y: [x, y], ages, vals))
        res_list.append([i, age_val])
    return res_list

def get_Umap_color(attr):
    return list(ordered_feats[attr])

def get_pat_demo(id):
    return ckd_crf_demo[ckd_crf_demo['pat_id']==id]

def get_df_concept(att_name):
    return ckd_data_df[ckd_data_df['concept.cd']==att_name]['nval.num']

def get_df_all_pat():
    # list =  [51475, 56237, 39463, 40545, 66708, 49217, 42882, 45666, 30475, 54265, 54539,
    #         65828, 41090, 49376, 42747, 50682, 56694, 25955, 58960, 28308, 47413, 65441,
    #         67438, 66718, 47978, 46873, 34422, 30620, 65278, 35921, 27387, 52906, 67311,
    #         53010, 32829, 19278]
    list = [42747, 41420, 59450, 54526, 55065, 59864, 58426, 57410, 66649, 63708, 52041, 51070, 49639, 16896, 66427, 59758, 58838, 52279, 57687, 62480, 41480, 54709, 42847, 58253, 61383, 41067, 67493, 65507, 63893, 50582, 65567, 41090, 60756, 59176, 45218, 59113, 61078, 58314, 65373, 52548, 66970, 38074, 32460, 63808, 41456, 56880, 64415, 63021, 65440, 37784, 60457, 63247, 53606, 64782, 42039, 54258, 58030, 44895, 58563, 24992, 66436, 38249, 65044, 64204, 60362, 63336, 29502, 52576, 27063, 33219, 54529, 46134, 63110, 59704, 49376, 66789, 67475, 48202, 66411, 54273, 48590, 61577, 53355, 66676, 47894, 41248, 57581, 66422, 44292, 58419, 51188, 47002, 38605, 56341, 52910, 65502, 34980, 49010, 55992, 50682, 64685, 36834, 51357, 35625, 47633, 45183, 39552, 33124, 58792, 66412, 28956, 44618, 65058, 53212, 65406, 63244, 43712, 32584, 26755, 42186, 32696, 65197, 45464, 58880, 60603, 41714, 55068, 52212, 51479, 61076, 66609, 41691, 48241, 52793, 38227, 60869, 40279, 56616, 48511, 56921, 42300, 60644, 42195, 57667, 66614, 61303, 42747, 42565, 47052, 47744, 66923, 53071, 63017, 64588, 39037, 54786, 30006, 56891, 65563, 43539, 59601, 34014, 65065, 50737, 57433, 39633, 40123, 55550, 57224, 48965, 35905, 50760, 29956, 67448, 45155, 51640, 42020, 40737, 52629, 51475, 42883, 61598, 56590, 57106, 64801, 56654, 45911, 66432, 64293, 39790, 66057, 67357, 55503, 36647, 54025, 60569, 61322, 49111, 61875, 26408, 53771, 57002, 64409, 31626, 45078, 42838, 53017, 28693, 46269, 34752, 41276, 63360, 66774, 42151, 53918, 40931, 34574, 51341, 34148, 61593, 39463, 61538, 47550, 42133, 40785, 40555, 60427, 37952, 52994, 48145, 32257, 46890, 56178, 35045, 44672, 40260, 44429, 37731, 43197, 43107, 62056, 25036, 28272, 23893, 60560, 50936, 46101, 60327, 44645, 59659, 52622, 65330, 63646, 39727, 58493, 51566, 48113, 37524, 40296, 42908, 61611, 61059, 66216, 51968, 58129, 54325, 31104, 37023, 55947, 28614, 28308, 60001, 36864, 65257, 40792, 53463, 37107, 53683, 36231, 52783, 35730, 39377, 16011, 42053, 39596, 39606, 42041, 30541, 17183, 49557, 64208, 65991, 41421, 19164, 55230, 20916, 29884, 43293, 60339, 46663, 66225, 55893, 39867, 59977]
    return list
    # return list(ckd_data_df['pat.id'].unique())

def get_df_all_concept():
    return list(ckd_data_df['concept.cd'].unique())

def get_pat_age_concept_list(pat_id):
    df = get_pat_records(pat_id)
    df = df.sort_values(by=['age'])
    concepts = df['concept.cd'].unique()
    ages = toIntergers(df['age'].unique())
    return list(range_list(ages[0], ages[-1]+1)), list(concepts)

def getLabTestViewData(pat_id):
    ages, pat_concepts = get_pat_age_concept_list(pat_id)
    concepts = get_df_all_concept()
    concept_age_val = get_pat_unique_concept(pat_id)
    res = []
    for concept_pair in concept_age_val:
        concept_ind = concepts.index(concept_pair[0])
        age_num_val_dict = { }
        for age_val_pair in concept_pair[1]:
            age = int(age_val_pair[0])
            val = age_val_pair[1]
            if age not in age_num_val_dict.keys():
                age_num_val_dict[age] = [1, val]
            else:
                pre = age_num_val_dict.get(age)
                val_max = val if val>pre[1] else pre[1]
                age_num_val_dict[age] = [pre[0]+1, val_max]
        for key in age_num_val_dict:
            age_ind = ages.index(key)
    #         print(age_ind, concept_ind, age_num_val_dict[key][0], age_num_val_dict[key][1])
            res.append([age_ind, concept_ind, age_num_val_dict[key][0], age_num_val_dict[key][1]])
    return res

look_up_p = pd.read_csv('data/look_up_p.csv')
def getIndicatorMarkers(pat_id):
    pat_df = features_all_csn[features_all_csn['pat_id']==pat_id].sort_values('age')
    ages, pat_concepts = get_pat_age_concept_list(pat_id)
    concepts = getOrderofConcepts(pat_id)
    my_set = set()
    res = []
    for age in ages:
        one_age_visit = pat_df[(pat_df['age']>age) & (pat_df['age']<age+1)]
        if len(one_age_visit) > 0:
            for index, row in one_age_visit.iterrows():
                label = row['cluster_label']
                group = label_category(label)
                mark = None
                if group is not None:
                    if group == 'before_1' or group == 'before_2':
                        mark ='predict'
                    else:
                        mark ='marker'
                    table_look_up = look_up_p[look_up_p['group']==group]
                    for var in table_look_up.var_name:
                        # var_val = row[var]
                        var_val = one_age_visit[var].mean()
                        ind_df = table_look_up[table_look_up['var_name']==var]
                        mean_x = list(ind_df.mean_x)[0]
                        mean_y = list(ind_df.mean_y)[0]
                        color_x = list(ind_df.color_1)[0]
                        color_y = list(ind_df.color_2)[0]
                        concept_1 = var[:-4]
                        if row[concept_1+'_avail']==1 and list(ind_df.p_val)[0]<0.05:
                            color_1 = None
                            if (ages.index(int(row.age)), concepts.index(concept_1)) in my_set:
                                continue
                            else: 
                                my_set.add((ages.index(int(row.age)), concepts.index(concept_1)))
                            if var_val>((mean_x+mean_y)/2):
                                color_1 = color_x if list(ind_df.stat)[0]>0 else color_y
                                res.append([ages.index(int(row.age)), concepts.index(concept_1), color_1, mark])
                            else:
                                color_1 = color_x if list(ind_df.stat)[0]<0 else color_y
                                res.append([ages.index(int(row.age)), concepts.index(concept_1), color_1, mark])
    return res

after_orange_after_green_blue = [49, 85, 72, 34, 25, 10, 33, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 0, 99]
after_green_after_blue = [60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54, 0, 99, 58, 29, 47, 82, 67, 14]
before_1 = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16]
before_2 = [89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33]

def label_category(label):
    if label in before_1:
        return 'before_1',
    if label in before_2:
        return 'before_2'
    if label in after_green_after_blue:
        return 'after_green_after_blue',
    if label in after_orange_after_green_blue:
        return 'after_orange_after_green_blue'
    return None

def getLabTestNormalData(pat_id):
    ages, pat_concept = get_pat_age_concept_list(pat_id)
    # concepts = get_df_all_concept()
    concepts = getOrderofConcepts(pat_id)
    concept_age_val = get_pat_unique_concept(pat_id)
    res = []
    for concept_pair in concept_age_val:
        concept_ind = list(concepts).index(concept_pair[0])
        concept_val = concept_pair[0]
        age_num_val_dict = { }
        for age_val_pair in concept_pair[1]:
            age = int(age_val_pair[0])
            val = age_val_pair[1]
            if age not in age_num_val_dict.keys():
                age_num_val_dict[age] = [0, 0, -9999, 0, 9999]
            low = normal_range_dict[concept_val][0]
            high = normal_range_dict[concept_val][1]
            pre = age_num_val_dict.get(age)
            if val > high:
                val_max = val if val>pre[2] else pre[2]
                age_num_val_dict[age] = [pre[0], pre[1]+1, val_max, pre[3], pre[4]]
            elif val < low:
                val_min = val if val<pre[4] else pre[4]
                age_num_val_dict[age] = [pre[0], pre[1], pre[2], pre[3]+1, val_min]
            else:
                age_num_val_dict[age] = [pre[0]+1, pre[1], pre[2], pre[3], pre[4]]
        for key in age_num_val_dict:
            age_ind = list(ages).index(key)
            values = age_num_val_dict[key]
    #         print(age_ind, concept_ind, age_num_val_dict[key][0], age_num_val_dict[key][1])
            res.append([age_ind, concept_ind, values[0], values[1], values[2], values[3], values[4], concept_val])
    return res

def toIntergers(data):
    return np.trunc(data).astype(int)

def range_list(a, b): 
    return list(range(a, b+1))

normal_range_dict = {
    'EGFR':[60, 200],
    'TBIL': [0.1, 1.2],
    'BP_DIASTOLIC': [60, 80],
    'BP_SYSTOLIC': [90, 120],
    'WT': [90, 220],
    'HT': [57, 78],
    'CHOLESTEROL': [50, 200],
    'CREATINE_KINASE': [22, 198],
    'HEMOGLOBIN': [11.6, 17.2],
    'INR': [0.8, 1.1],
    'ALT_SGPT': [7, 56],
    'AST_SGOT': [8, 45],
    'ALK': [44, 147],
    'HDL': [40, 100],
    'LDL': [40, 100],
    'TRIGLYCERIDES': [20, 150],
    'HBA1C': [4, 6.5],
    'TROPONIN': [0, 0.04]
}

def getHierarchicalClusterVec(pat_id):
    ages, concepts = get_pat_age_concept_list(pat_id)
    concept_age_val = get_pat_unique_concept(pat_id)
    concept_vec_dict = {}
    for concept_pair in concept_age_val:
        concept_ind = concept_pair[0]
        age_num_val_dict = {}
        for age_val_pair in concept_pair[1]:
            age = int(age_val_pair[0])
            val = age_val_pair[1]
            if age not in age_num_val_dict.keys():
                age_num_val_dict[age] = [1, val]
            else:
                pre = age_num_val_dict.get(age)
                val_max = val if val>pre[1] else pre[1]
                age_num_val_dict[age] = [pre[0]+1, val_max]
        res = []
        for age_uni in ages:
            age_ind = list(ages).index(age_uni)
            if age_uni in age_num_val_dict.keys():
                # res.append(age_num_val_dict[age_uni][0])
                res.append(1)
            else: res.append(0)
        concept_vec_dict[concept_ind]=res
    return concept_vec_dict


def getHierarchicalClusterInput(pat_id):
    matrix = []
    vect_dict = getHierarchicalClusterVec(pat_id)
    for key in vect_dict:
        matrix.append(vect_dict[key])
    return matrix

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def getOrderofConcepts(pat_id):
# setting distance_threshold=0 ensures we compute the full tree.
    matrix = getHierarchicalClusterInput(pat_id)
    res = None
    ages, concepts = get_pat_age_concept_list(pat_id)
    concepts_all = get_df_all_concept()
    if len(matrix)==1:
        res = [concepts[0]]
    else:
        model = AgglomerativeClustering(linkage="ward", distance_threshold=2, n_clusters=None)
        labels = model.fit_predict(matrix)
        concepts = np.array(concepts)
        key_tuples = []
        for i in range(len(concepts)):
            key_tuples.append(new_key(concepts[i], labels[i]))
        newlist = sorted(key_tuples, key=lambda x: x.label, reverse=True)
        res = [i.name for i in newlist]
        # res = list(concepts[labels])
    for i in concepts_all:
        if i not in res:
            res.append(i)
    return res


class new_key:
      def __init__(self, name, label):
        self.label = label
        self.name = name