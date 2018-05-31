from sklearn.externals import joblib
import context_dictionary as ctxt
import json
from apiTools.util import *

context_dic =  joblib.load('./results/viavoo_viavoo_RSE_ControlParental_3mois_G0R5_20180403104341.data')

# list of kernels
print("_______kernels________")
print(context_dic.kernels[0:10])





# dict{key:kernel,value:score}
print("_______scores________")
#for k in context_dic.scores:
#    print(k,context_dic.scores[k])




# dict{} !!!!! its empty!!!!
print("_______clusters_________")
i = 0
for k in context_dic.clusters:
    if i<2:
        print(k, context_dic.clusters[k])
        i+=1
    else:
        break



# dict{key:kernel,value: to ask!!!!!!!} 
print("_______ctxt_dict________") 
'''
key(string): kernel enrichi
value(list of tuple):syntagme, times that syntagme appear(int),  verbatime indexs(set(int)), clusters(string)
'''
i = 0
for k in context_dic.ctxt_dict:
    if i<2:
        print(k)
        print(context_dic.ctxt_dict[k])
        print(list(context_dic.ctxt_dict[k][0][2]))
        i+=1
    else:
        break


# foucus: dict{famille:[key words for famille], device:[keywords for device]}

print (json.dumps(context_dic.documents,separators=(',',':'),ensure_ascii=False))