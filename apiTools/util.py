"""
format kernel1
{kernel1:String,score:Number}
"""
def formatKernel1s(scores):
    res  = list()
    for k in scores:
        elem = dict()
        elem['kernel1'], elem['score'] = k, scores[k]
        res.append(elem)
    return res

"""
format kernel2
exemple
{
    "kernel2": "numéro de pervers",
    "phrases": [
        {
            "phrase": "ce est sosh qui me donne un numéro de pervers",
            "appearTimes": 1,
            "verbatimes": [
                {
                    "idx": 534,
                    "tags": [
                        "device"
                    ]
                }
            ],
            "cluster": "K0"
        }
    ]
}
"""
def formatKernel2s(k2dict,docs):

    def verbatimIdxWithTags(vlist):
        if vlist:
            res = list()
            for i in vlist:
                elem = dict()
                elem['idx'], elem['tags']=i, set2list(docs[i]['crossTags'])
                res.append(elem)
            return res
        print('verbatime list is empty!')
        return None

    res = list()
    for k in k2dict:
        elem = dict()
        elem['kernel2'] = k
        elem['phrases'] = list()
        for p in k2dict[k]:
            el = dict()
            el['phrase'],el['appearTimes'],el['verbatimes'],el['cluster'] = p[0],p[1],verbatimIdxWithTags(p[2]),p[3]
            elem['phrases'].append(el)
        res.append(elem)
    return res

'''make verbatim indexes in k2dict the list indexes of docs, then delete the docs idx attribute, in this way, a verbatime can be accessible par a list index, its faster'''
def ressovleIndexProb(docs,k2dict):

    def findDocIdx(n):
        for i in range(len(docs)):
            if docs[i]['idx']==n:
                return i
        print('index '+ str(n) +' did not find in docs')
        return None

    for k in k2dict:
        k2dict[k] = [list(p) for p in k2dict[k]]
        for j in range(len(k2dict[k])):
            k2dict[k][j][2]=[findDocIdx(i) for i in k2dict[k][j][2]]
    
    for doc in docs:
        del doc['idx']

    return docs, k2dict
        
'''add tags in the verbatime element. This function takes a list of verbatim index, return a list of verbatim objects with tags'''
def verbatimIdxWithTags(vlist, docs):
    if vlist:
        res = list()
        for i in vlist:
            elem = dict()
            elem['idx'], elem['tags']=i, set2list(docs[i]['crossTag'])
            res.append(elem)
        return res
    print('verbatime list is empty!')
    return None


def findK2sByK1(k1,k2s):
    res = list()
    for k2 in k2s:
        if k1 in k2['kernel2']:
            res.append(k2)
    return res 

def findK2sByTags(tags,k2s):
    res=list()
    # to add
    return res     
            
def str2idxList(s):
    return [int(n) for n in s.split(',')]


def set2list(s):
    res = list()
    if s:
        for e in s:
            res.append(e)
    return res



    