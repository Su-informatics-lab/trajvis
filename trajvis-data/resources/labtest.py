from flask_restful import Resource
import sys

sys.path.append("..")
from services.get_df_data import getLabTestNormalData, getOrderofConcepts, get_pat_age_concept_list,getIndicatorMarkers

class Labtest(Resource):
    def get(self, id):
        ages, pat_concepts = get_pat_age_concept_list(id)
        concepts = getOrderofConcepts(id)
        data = getLabTestNormalData(id)
        marker = getIndicatorMarkers(id)
        return {
            'ages': ages,
            'concepts': concepts,
            'data': data,
            'marker': marker
        }
    

