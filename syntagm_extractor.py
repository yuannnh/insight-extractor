# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:06:14 2017

@author: KHZS7716
"""
#%% inits
import settings as env
env.init()

wv_models = {'wiki': "wiki.fr.vec.bin",
             'viavoo': "orange.fr.vec.bin",
             'radarly': "bds.orange.fr.vec.bin"}

#%% imports
import sys
import os
import re
import json
import time

import pandas as pd
import numpy as np

from sklearn.externals import joblib
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from gensim import corpora, models, similarities
from treetaggerwrapper import Tag

import commons as com
import context_dictionary as ctxt
import nltk_vars as vars

#%% lof file function
def log(message, file=env.out):
    log_msg = "".join((time.strftime("%Y-%m-%d %H:%M:%S")," : ", message))
    print(log_msg, file = file)
    if not env.debug : print(log_msg)

#%% unique_docs : supprime les doublons parfaits dans le corpus
def unique_docs(docs):
    log("Suppression des doublons parfaits dans le corpus...")
    len_start = len(docs)
    unique_docs = list()
    for doc in docs:
        if doc not in unique_docs: unique_docs.append(doc)
    log("{} doublons supprimés.".format(len_start-len(docs)))
    
    return(unique_docs)

#%% loadDocuments : trabnsforme le corpus dans un structure Document
def loadDocuments(source, datafile, stopwords=vars.my_stopwords, rm_duplicates=True):
    all_documents, docs = list(), list()

    if source == 'txt':
        # connecteur pour fichier text sans header - 1 ligne de verbatim
        with open(datafile, 'r', encoding='utf-8') as f:
            for line in f:
                docs.append((os.path.basename(datafile),'no date', com.preprocessing_raw(line))) #get the cleaned raw line
    
    elif source == 'webcrawl':
        # connecteur pour [(domain,url,time,[text])]
        with open(datafile, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
    elif (source == 'viavoo'):
        data=pd.read_excel(datafile,header=0,skiprows=4)
        docs = [ (os.path.basename(datafile), line['Date'], com.preprocessing_raw(line['Texte'])) for line in data ]
        
    elif (source == 'viavoo_hd'):
        data=pd.read_csv(datafile, sep='\t', header=0)
        docs = [ (os.path.basename(datafile), row['date_value'], com.preprocessing_raw(row['texte'])) for _,row in data.iterrows() ]
        
    elif (source == 'oem'):
        # format Orange et Moi (dmgp) pour insatisfaction
        data=pd.read_excel(datafile, header=None,skiprows=3)
        for text in data[0]:
            text = re.sub('^ - ','',text) # traitement specifique à cette source
            docs.append((os.path.basename(datafile),'no date', com.preprocessing_raw(text))) #initialize an empty corpus
    
    elif (source == 'cindy'):
        # format Orange et Moi (cindy Wooh) pour insatisfaction
        sheets = ['Ergonomie','fonctionnement','Contenu','Contacts SC']
        for sheet in sheets:
            data=pd.read_excel(datafile, sheet_name=sheet, header=None, skiprows=3, usecols=[0])
            for text in data[0]:
                text = re.sub('^ - ','',text) # traitement specifique à cette source
                docs.append((sheet,'no date', com.preprocessing_raw(text))) #initialize an empty corpus

    elif (source == 'radarly'):
        with open(datafile,'r',encoding='utf-8') as f:
            header = f.readline()
        headers = header[:-1].replace('"','').split(';')
        dtype_dic = dict.fromkeys(headers,object)
        
        data = pd.read_csv(datafile, sep=';', header=0, quotechar='"', dtype=dtype_dic)
        data = data[['site name','date','text','lang']]
        data = data[data['text'].notnull()]
        docs = [ (line[1]['site name'], line[1]['date'], com.preprocessing_raw(line[1]['text'])) for line in data.iterrows() if line[1]['lang'] == 'fr' ]
    
    # traitements communs
    if rm_duplicates: docs = unique_docs(docs)
    all_documents = com.get_documents(docs, stopwords)
    log("Fin de chargement des documents")

    return all_documents

#%% tech vars
wScreen = 90 # line width for repeat_str
grammar_path ='./grammars'

#%% extracteur de syntagmes
def get_insight(tags, search_noun, min_len):
    """get context -1 +1 of a particular noun"""
    noun_ctxs=list()
    tokens=[tok.word for tok in tags] # get word tokens
    idxs=[i for i,tag in enumerate(tags) if search_noun in tag.lemma ]
    for idx in idxs:
        before=tags[0:idx] # tag list before search noun
        before.reverse() # reversed tag list
        after=tags[idx+1:] # tag list after search noun
        beg,end=str(),str()
        if len(before)>0:
            for i,tag in enumerate(before):
                if tag.pos not in {'KON', 'PUN', 'PUN:cit', 'SENT'}:
                    beg=' '.join((tag.word,beg))
                else:
                    break
        if len(after)>0:
            for i,tag in enumerate(after):
                if tag.pos not in {'KON', 'PUN', 'PUN:cit', 'PRO:PER', 'SENT'}:
                    end=' '.join((end,tag.word))
                else:
                    break
        syntagm = beg+tokens[idx]+end
        syntagm_nwords = len(syntagm.split())
        if syntagm_nwords > min_len and syntagm_nwords < 20: noun_ctxs.append(syntagm)
    return noun_ctxs

#%%
def word_clustering(words, model, vocab, nK=12):
    """Recalcule les clusters du dictionnaire en se basant sur le vocabulaire enrichi"""
    kernels = np.array([ k for k in words if k in vocab ])
    data = model[kernels]
    kmeans = KMeans(n_clusters=nK, init='k-means++').fit(data)
    labels = np.array(kmeans.labels_)
    
    res=dict()
    
    for k in range(nK):
        filtre = np.array(labels == k)
        kKernels = kernels[filtre]
        titres = [ k[0] for k in res.values() ]
        if len(kKernels) == 1:
            tmp_titles = kKernels
        else :
            tmp_titles = [ k[0] for k in model.most_similar(positive=kKernels[0:5], topn=10) if k[0] in vocab ]
        # boucle sur les titres temporaires 
        for title in tmp_titles:
            if title not in titres:
                kTitle = title
                break
        
        titres.append(kTitle)
        res.update({k:(kTitle,kKernels)})
    
    return res

#%% perform syntagms similarity
def synt_similarity(syntagms):
    """returns syntagms similarity"""
    texts = [[word for word in synt[0].split() if word not in vars.my_stopwords] for synt in syntagms]
    
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token]+=1
    texts = [[token for token in text if frequency[token] > 1] for text in texts ]
    dictionary = corpora.Dictionary(texts)
    
    # doc2bow counts the number of occurences of each distinct word,
    # converts the word to its integer word id and returns the result
    # as a sparse vector
    
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    
    synt_sims = list()
    for doc in texts:
        vec_bow = dictionary.doc2bow(doc)
        # convert the query to lsi space
        vec_lsi = lsi[vec_bow]
        index = similarities.MatrixSimilarity(lsi[corpus])
        # perform a similarity query against the corpus
        sims = index[vec_lsi]
        synt_sims.append(sims)
        
    return synt_sims

#%%
def print_clusters(clusters):
    """Affiche les clusters internes du dictionnaire
    :param clusters: { id_cluster int : (titre str, [noyaux str]) }"""
    print("Macro thèmes :")
    for k in clusters:
        line = "Cluster {} = {} ({} noyau(x))".format(k, clusters[k][0], len(clusters[k][1]))
        print(line)
    input("-- suite --")
    print(com.repeat_str("-", wScreen))
    print('Détail des clusters :')
    for k in clusters:
        line = "Cluster {} = {} :\n{}\n".format(k, clusters[k][0], clusters[k][1])
        print(line)
        input("-- suite --")

#%% test clustering
def aggK(syntagms):
    tol = 1e-4
    seuil = 0.1
    
    sims = np.array([[doc for doc in sim] for sim in synt_similarity(syntagms)])
    
    inertias = list()
    inertia_gaps = list()
    for nk in range(1,len(syntagms)):
#        print("Test KMeans nk",nk)
        inertia = KMeans(n_clusters=nk, tol=tol).fit(sims).inertia_
        inertias.append(inertia)
        if len(inertias) > 1 :
            gap = inertias[-2]-inertias[-1]
            inertia_gaps.append(gap)
            if gap < seuil : break
    
#    x = [ i+1 for i,_ in enumerate(inertias)]
#    plt.plot(x,inertias)
#    plt.ylabel('K inertia')
#    plt.xlabel('nK')
#    #plt.xticks(x)
#    plt.show()
    
    for idx,dist in enumerate(inertia_gaps):
        if dist < seuil:
            nK = idx+1
            break
    
    log("Nb syntagms : "+str(len(syntagms)))
    log("Best nK : "+str(nK))
    log("Taux de compression des syntagmes : {:.2f}%".format((1-nK/len(syntagms))*100))
    
    kmeans = KMeans(n_clusters=nK).fit(sims)
    
    return({ik:[i for i,k in enumerate(kmeans.labels_) if k == ik] for ik in range(nK)})

#%% utility functions : get_context_dict
def get_context_dict(documents, lscore=0, focus=None, rm_duplicates=True, min_len=3):
    """get the context_dict from documents
    
    :param nkernel: <450>, nombre max de noyaux conservés dans le dictionnaire
    :param focus: <None>, dictionnaire (theme: [patterns]) des themes focus à extraire
    :param rm_duplicates: <True>, supprime les doublons du corpus initial
    :param min_len: <3>, longueur min des syntagmes remontés
    
    :return: context dictionary (type = context_dictionary)
    """
    noun_pos = { 'NOM', 'NAM', 'ABR' }
    
    # remove duplicates
    if rm_duplicates:
        log("Suppression des doublons dans le corpus :")
        len_start = len(documents)
        documents = com.remove_duplicates(documents)
        log("{} doublons supprimés.".format(len_start-len(documents)))
    
    # init context_dictionary
    ctd = ctxt.context_dictionary(documents)
    
    # extraction des phrases
    log('Extraction des phrases...')
    doc_sentences = list()
    for doc in documents:
        phrases = com.get_phrases(doc['tags'])
        if phrases:
            doc_sentences.append((doc['idx'],phrases))
    
    # traitement des ngrams telcos
    log('Traitement des ngrams telcos...')
    doc_sentences = [ tfind(ds, vars.telcoms) for ds in doc_sentences ]
    
    # extraction des lemmes de noyaux nominaux
    log('Extraction des lemmes de noyaux nominaux...')
    words_dic = { tag.lemma:tag.word for doc_sen in doc_sentences for s in doc_sen[1] for tag in s if tag.pos in noun_pos and tag.word not in vars.my_stopwords and tag.lemma not in vars.stoplemmas and len(tag.lemma) > 1 }
    lemma_nouns = set(words_dic.keys())
    
    # Extraction du dictionnaire des noms
    log('Extraction du dictionnaire des noms :')
    idx, start_time = 1, time.time() # init progress_per
    lemma_dic = dict.fromkeys(lemma_nouns)
    for noun in lemma_nouns:
        noun_doc_sent = list()
        for doc_sen in doc_sentences:
            doc_sent = list()
            for s in doc_sen[1]:
                if noun in [t.lemma for t in s if t.pos in noun_pos]:
                    doc_sent.append(s)
            if doc_sent: noun_doc_sent.append((doc_sen[0], doc_sent))
        lemma_dic[noun] = noun_doc_sent
        idx = com.progress_per(idx, len(lemma_nouns), start_time) # print the progress percentage info
    print()
    
    # fréquence pour le lemma : nombre de documents différents et nombre de phrases dans chaque doc
    # score dépendant du nombre de docs et du nombre de phrases dans le doc
    #scores = { n:len(lemma_dic[n])*(1+np.log(sum([len(d[1]) for d in lemma_dic[n]]))) for n in lemma_dic }
    # score TFDF : fréquence nb de docs x nb de phrases total
    #scores = { n:len(lemma_dic[n])*sum([len(d[1]) for d in lemma_dic[n]]) for n in lemma_dic }
    # score TF : term frequency
    scores = { n:sum([len(d[1]) for d in lemma_dic[n]]) for n in lemma_dic }
    
    # extraction du nombre de noyaux demandés, triés par score
    kernels = [ s for s in com.sort_dictionary(scores) if s[1] > lscore ]
    score_max = kernels[0][1]
    for k in list(scores): scores[k] = scores[k]/score_max # score normalization
    ctd.scores = { k[0]: scores[k[0]] for k in kernels } # score filter
    kernels = { k[0] for k in kernels }
    
    # traitement des adverbes avec la grammaire gADV
    log('Extraction des adverbes avec la grammaire gADV...')
    grammar = load_grammar('gADV.txt', grammar_path)
    doc_sentences = [ sfind(ds, grammar, 'ADV') for ds in doc_sentences ]
    
    # traitement des adjectifs avec la grammaire gADJ
    log('Extraction des adjectifs avec la grammaire gADJ...')
    grammar = load_grammar('gADJ.txt', grammar_path)
    doc_sentences = [ sfind(ds, grammar, 'ADJ') for ds in doc_sentences ]
    
    # traitement des groupes verbaux au présent avec la grammaire gVERpres
    log('Extraction des groupes verbaux au présent avec la grammaire gVERpres...')
    grammar = load_grammar('gVERpres.txt', grammar_path)
    doc_sentences = [ sfind(ds, grammar, 'VER:pres') for ds in doc_sentences ]
    
    # traitement des groupes verbaux à l'infinitif avec la grammaire gVERinfi
    log("Extraction des groupes verbaux à l'infinitif avec la grammaire gVERinfi...")
    grammar = load_grammar('gVERinfi.txt', grammar_path)
    doc_sentences = [ sfind(ds, grammar, 'VER:infi') for ds in doc_sentences ]
    
    # traitement des groupes nominaux avec la grammaire gNOM
    log('Extraction des groupes nominaux avec la grammaire gNOM...')
    grammar = load_grammar('gNOM.txt', grammar_path)
    doc_sentences = [ sfind(ds, grammar, 'NOM') for ds in doc_sentences ]
    
    # Nouvelle extraction des lemmes de syntagmes nominaux
    log('Extraction des lemmes de syntagmes nominaux...')
    words_dic = { tag.lemma:tag.word for doc_sen in doc_sentences for s in doc_sen[1] for tag in s if tag.pos in noun_pos and len(kernels.intersection(tag.lemma.split())) > 0 }
    synt_nouns = set(words_dic.keys())
    ctd.kernels = list(synt_nouns)
    
    # Extraction du dictionnaire des noms
    log('Extraction du dictionnaire des syntagmes :')
    idx, start_time = 1, time.time() # init progress_per
    synts_dic = dict.fromkeys(synt_nouns)
    for noun in synt_nouns:
        noun_doc_sent = list()
        for doc_sen in doc_sentences:
            doc_sent = list()
            for s in doc_sen[1]:
                if noun in [t.lemma for t in s]:
                    doc_sent.append(s)
            if doc_sent: noun_doc_sent.append((doc_sen[0], doc_sent))
        synts_dic[noun] = noun_doc_sent
        idx = com.progress_per(idx, len(synt_nouns), start_time) # print the progress percentage info
    print()

    non_signifiant = list() # init liste des noyaux sans syntagmes signifiants

    # Boucle sur les noayux - calcul des syntagmes
    ##############################################
    log('Extraction des syntagmes...')
    
    for lemma in synts_dic: # boucle sur les noyaux
        ctxs,idxs,context_list=list(),list(),list()
        # calcul des contextes par lemme dans le corpus corrigé
        for doc_s in synts_dic[lemma]:
            for sentence in doc_s[1]:
                if words_dic[lemma] in [t.word for t in sentence] :
                    noun_ctx=get_insight(sentence,lemma,min_len) # liste des contextes du mot dans la phrase
                    ctxs.extend(noun_ctx) # maj liste des contextes dans tous les documents
                    idxs.extend([doc_s[0]]*len(noun_ctx)) # maj liste de même taille contenant l'index du document répété pour chaque contexte
        nb_ext = len(ctxs)
        
        # agrégation des contextes
        grouped_ctxs = com.sort_dictionary(Counter(ctxs))
        
        # loop on unique context
        for ctx in grouped_ctxs:
            grouped_idxs=set()
            # tant que trouve le ctx dans la liste des ctxs
            while ctx[0] in ctxs:
                i=ctxs.index(ctx[0]) # retrieve index du contexte dans la liste
                grouped_idxs.add(idxs.pop(i)) # ajoute l'index du doc pour ce contexte a la liste groupée des index unique de documents
                ctxs.pop(i) # supprime le ctx de la liste des ctxs
            context_list.append((ctx[0],ctx[1],grouped_idxs))
        nb_uni = len(grouped_ctxs)
        
        # agrégation des syntagmes
        len1 = len(context_list) # longueur de départ de la liste des syntagmes
        for i,tup in enumerate(context_list):
            for j,tup2 in enumerate(context_list):
                if tup and tup2 and j != i and tup[0] in tup2[0]:
                    context_list[i] = (tup[0],context_list[i][1]+tup2[1],context_list[i][2].union(tup2[2]))
                    context_list[j] = None
        context_list = [ tup for tup in context_list if tup ]
        context_list.sort(key=lambda x: x[1], reverse=True) # Tri des syntagmes par nb occurences
        nb_aggr = len1 - len(context_list)
        
        # syntagm similarity if more than 13
        if len(context_list) > 13:
            log("Calcul Group Similarity des syntagmes de : "+lemma)
            clust = aggK(context_list)
            for k in clust:
                for id in clust[k]:
                    context_list[id] = (context_list[id][0], context_list[id][1], context_list[id][2], 'K'+str(k))
        else :
            context_list = [(tup[0], tup[1], tup[2], 'K0') for tup in context_list]
        
        log("Noyau [{}] : {} validé(s), {} agrégé(s), {} unique(s), {} extrait(s).".format(lemma, len(context_list), nb_aggr, nb_uni, nb_ext))
        
        if len(context_list)>0:
            ctd.ctxt_dict.update({lemma:context_list})
        else:
            non_signifiant.append(lemma) # liste des noyaux sans syntagmes signifiants
    
    # mise à jour de la liste des noyaux signifiants
    log('Liste des noyaux sans syntagmes signifiants : '+str(non_signifiant))
    ctd.kernels = [ k for k in ctd.kernels if k not in non_signifiant ]
    
    # get focus if given
    # FIXME le focus par mots-clés est inséré ici car la phase de création des docs est longue
    log('Pose des tags focus sur les documents...')
    if focus:
        ctd.focus = focus
        com.get_focus(documents, focus)

    return ctd

#%%
def prep(s, rule, posTag, form=1):
    """ retourne la structure imbriquée [ phrases [ règle syntaxik [ Tags ]]] de départ en remplaçant le tag par le syntagme
    form : 0 = word, 1 = pos, 2 = lemma"""
    pattern, lem_seq = rule[0], rule[1]
    res = list()
    seq = [ t[form] for t in s ] # flatten form : 0 = word, 1 = pos, 2 = lemma
    l = len(pattern)
    idx=0
    while idx < len(seq):
        tag = s[idx]
        if tag[form] == pattern[0] :
            if len(seq[idx:]) >= l:
                ok = True
                for i,j in enumerate(range(idx, idx+l)):
                    ok = ok & (s[j].pos == pattern[i])
                if ok :
                    res.append(Tag(word=" ".join([t.word for t in [s[i+idx] for i in range(l)]]), pos=posTag, lemma=" ".join([t.lemma for t in [s[i+idx] for i in lem_seq]])))
                    idx+=l
                else :
                    res.append(tag)
                    idx+=1
            else :
                res.append(tag)
                idx+=1
        else :
            res.append(tag)
            idx+=1

    return res

#%%
def load_grammar(file, grammar_path):
    g = list()
    with open('/'.join((grammar_path,file)), 'r') as f:
        for line in f:
            if line[0] != '#':
                row = line.split('\t')
                if len(row) == 2:
                    g.append([eval(row[0]),eval(row[1])])
    return g

#%%
def sfind(doc_sentence, grammar, posTag):
    res = list()
    for sen in doc_sentence[1]:
        new_sen = sen.copy()
        for rule in grammar:
            new_sen = prep(new_sen,rule,posTag)
        res.append(new_sen)
    return (doc_sentence[0], res)

#%%
def tfind(doc_sentence, ngrams):
    res = list()
    for sen in doc_sentence[1]:
        new_sen = sen.copy()
        for ngram in ngrams:
            rule = [ (ngram.split(),[i for i,_ in enumerate(ngram.split())]) for ngram in vars.telcoms]
            new_sen = prep(new_sen,rule,'NOM',0)
        res.append(new_sen)
    return (doc_sentence[0], res)

#%% traitement
if __name__ == "__main__":
    assert len(sys.argv) == 3
    source = sys.argv[1]
    datafile = sys.argv[2]
    dest = os.path.splitext(os.path.basename(datafile))[0]
    
    dic_file = '/'.join((env.resultpath,'_'.join((source,dest,env.version,env.runtime))))
    
    log("Debut du traitement")
    log(" ".join(("Chargement des documents",source,os.path.basename(datafile))))

    #___create a list of verbatime, cf: test_result.json____
    #___{id,source,time,raw(verbatime),[word,pos,lemma]}___
    all_documents = loadDocuments(source, datafile)
    
    focus = {'famille':vars.avec_qui}
    
    log("Création du dictionnaire ...")
    context_dict = get_context_dict(all_documents, lscore=0, focus=None, rm_duplicates=True)
    if env.verbose:
        log("Ecriture pickle du dictionnaire context_dict...")
        joblib.dump(context_dict, dic_file+'.data')
        log("Ecriture txt des syntagmes du dictionnaire context_dict...")
        context_dict.write_context_dict(dic_file+'.txt', count=0)
    log("Export json pour analyse dans l'interface...")
    context_dict.exportJSON(dic_file)
    log("Fin de création du dictionnaire context_dict.")
    env.out.close()

