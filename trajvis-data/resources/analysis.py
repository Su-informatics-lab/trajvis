from flask_restful import Resource
import sys

sys.path.append("..")
from services.get_analysis import get_concept_distribution,get_three_uncertainty, getThreePossibility, getTrajectoryPoints, get_pat_race_distribution, get_one_pat_cluster_label_by_age, get_pat_sex_distribution, fakeBlueData, fakegreenData, fakeOrangeData

class Analysis(Resource):
    def get(self, id):
        trajs = getTrajectoryPoints()
        age, ages, green_width, blue_width, orange_width = get_one_pat_cluster_label_by_age(id)
        ages_unc, green_unc, blue_unc, orange_unc = get_three_uncertainty(id)
        ages, green_poss, blue_poss, orange_poss = getThreePossibility(ages, green_width, blue_width, orange_width)
        return {
            'traj': trajs,
            'age': age,
            'blue_area': fakeBlueData(ages_unc, blue_unc),
            'green_area': fakegreenData(ages_unc, green_unc),
            'orange_area': fakeOrangeData(ages_unc, orange_unc),
            'sex_dist': get_pat_sex_distribution(),
            'race_dist': get_pat_race_distribution(),
            'x_range': ages,
            'green_poss': green_poss,
            'blue_poss': blue_poss,
            'orange_poss': orange_poss
        }
    
class AnalysisDist(Resource):
    def get(self, concept):
        x_vals, y_vals = get_concept_distribution(concept)
        return {
            'x_vals': x_vals,
            'y_vals': y_vals
        }
    

