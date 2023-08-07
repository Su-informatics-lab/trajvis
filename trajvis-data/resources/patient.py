from flask_restful import Resource
from flask import make_response
import sys
import json
from flask import jsonify

sys.path.append("..")
from services.get_df_data import get_pat_records, get_df_all_pat, get_pat_demo, get_pat_unique_concept, get_profile_date

class Patient(Resource):
    def get(self, id):
        records = get_pat_records(id).sort_values(by="age")
        demo_info = get_pat_demo(id)
        concept_info = get_pat_unique_concept(id)
        
        if len(demo_info)>0:
            demo_info = demo_info.to_json(orient="records", force_ascii=False)
        if len(records)>0:
            records = records.to_json(orient="records", force_ascii=False)
        else:
            records = None
        return {
            'records': records,
            'demo':demo_info,
            'concept': concept_info,
            'date': get_profile_date(id)
        } 



class AllPatient(Resource):
    def get(self):
        return jsonify({ 
            "data": list(map(lambda x:str(x), get_df_all_pat()))
        })
