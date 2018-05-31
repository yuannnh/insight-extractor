# RESTFUL API FOR CONTEXT DICTINARY
from sklearn.externals import joblib
import context_dictionary as ctxt
from flask import Flask
from flask import request
from flask_cors import CORS
import json
from apiTools.util import *

app = Flask("api")
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# load pkl data
context_dic =  joblib.load('./results/viavoo_viavoo_RSE_ControlParental_3mois_G0R5_20180403104341.data')

####################################################################################
#                            FORMATER LES DONNEES
# Cette partie est pour formater les données qu'on extrait de pkl data afin de 
# adapter le format de données d'un api.
# documents est un liste de verbatime.
# kernel1s est un liste de élement 
documents, kernel2s = ressovleIndexProb(context_dic.documents,context_dic.ctxt_dict)#résoudre le problème de index pour ctxt_dict et documents
kernel1s = formatKernel1s(context_dic.scores)
kernel2s = formatKernel2s(kernel2s,documents)

@app.route('/')
def index():
    return 'Hello'


@app.route('/kernels')
def kernels():
    res = list()
    for k in kernel1s:
        elem = dict()
        print(k)
        elem = k
        elem['kernel2s'] = findK2sByK1(k['kernel1'],kernel2s)
        res.append(elem)
    return json.dumps(res,separators=(',',':'),ensure_ascii=False)

#/kernel1s (with scores)
@app.route('/kernel1s')
def kernel1():
    n = request.args.get('n')
    res = kernel1s
    if n: res = k[0:int(n)]
    return json.dumps(res,separators=(',',':'),ensure_ascii=False)

#/score/[kernelname] (not suggested)
@app.route('/score/<string:kernel>')
def score(kernel):
    res=dict()
    res['kernel'], res['score'] =kernel, context_dic.scores[kernel]
    return json.dumps(res,separators=(',',':'),ensure_ascii=False)

#/kernel2s?kernel1=[String](if param is absent, return all the kernel2s)
@app.route('/kernel2s')
def kernel2():
    kernel1 = request.args.get('kernel1')
    tags = request.args.get('tags')
    res = kernel2s
    if kernel1:
        res = findK2sByK1(kernel1,res)
    if tags:
        # to do later 
        pass
    return json.dumps(res,separators=(',',':'),ensure_ascii=False)

#/docs?indexes=[numbers seperated by ,](param required)
@app.route('/docs')
def docs():
    indexes = request.args.get('indexes')
    indexes = str2idxList(indexes)
    res = list()
    for i in indexes:
        res.append(documents[i]['raw'])
        #print(context_dic.documents[i])
    return json.dumps(res,separators=(',',':'),ensure_ascii=False)

