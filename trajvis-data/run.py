import importlib
from flask import Flask, Blueprint
from flask_restful import Api
from flask_cors import CORS

from resources.patient import Patient, AllPatient
from resources.indicator import Indicator, AllIndicator
from resources.umap import Umap, PatProj
from resources.labtest import Labtest
from resources.analysis import Analysis, AnalysisDist

app = Flask(__name__)
CORS(app)
api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(Patient, '/api/patient/<int:id>')
api.add_resource(AllPatient, '/api/patients')
api.add_resource(AllIndicator, '/api/indicators')
api.add_resource(Indicator, '/api/indicator/<att_name>')
api.add_resource(Umap, '/api/umap')
api.add_resource(Labtest, '/api/labtest/<int:id>')
api.add_resource(PatProj, '/api/umap/<int:id>')
api.add_resource(Analysis, '/api/analysis/<int:id>')
api.add_resource(AnalysisDist, '/api/analysis/dist/<concept>')


app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
