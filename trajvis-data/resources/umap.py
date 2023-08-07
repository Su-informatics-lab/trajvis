from flask_restful import Resource
import sys
import json
sys.path.append("..")

from services.get_umap import get_orginal_embed, get_four_trajectory, project_to_umap, get_pat_age_egfr
from services.get_df_data import get_Umap_color

class Umap(Resource):
    def get(self):
        embed = list(map(lambda x, y, z: [x[0], x[1], y, z], get_orginal_embed(), 
                get_Umap_color('age'), get_Umap_color('egfr')))
        return { 
            "embed": embed,
            "traj": get_four_trajectory()
        }

class PatProj(Resource):
    def get(self, id):
        ages, egfrs = get_pat_age_egfr(id)
        embed = list(map(lambda x, y, z: [x[0].item(), x[1].item(), y, z], project_to_umap(id), ages, egfrs))
        return {
            "embed": embed
        }