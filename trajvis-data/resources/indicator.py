from flask_restful import Resource
import sys
from flask import jsonify

sys.path.append("..")
from services.get_df_data import get_df_concept, get_df_all_concept

class Indicator(Resource):
    def get(self, att_name):
        records = get_df_concept(att_name)
        if len(records)>0:
            records = records.to_json(orient="records", force_ascii=False)
        else:
            records = None
        return records
    
class AllIndicator(Resource):
    def get(self):
        return get_df_all_concept()

