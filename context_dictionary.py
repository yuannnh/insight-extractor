# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:07:13 2017

@author: KHZS7716
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import commons as com

#%%
#!!! deprecated
def menu(actions, level=0, top=None):
    if level == 0: menu = [ a for a in actions if len(a[0]) == 1 ]
    else : menu = [ a for a in actions if len(a[0]) == level+1 and a[0][level-1] == str(top) ]
    for m in menu:
        print("{}: {}".format(m[0][level:],m[1]))
    choix = input('choix ?: ')
    
    return choix if len(choix)>0 else 'X'

#%%
#!!! deprecated
def validate_list(listin):
    boucle = True
    listout = list()
    print("Liste initiale: ".format(listin))
    while boucle:
        for i,item in enumerate(listin):
            if input("{} - Valider {} ? (o): ".format(i,item)) == 'o':
                listout.append(item)
        print("Liste finale: "+", ".join(listout))
        boucle = input("Valider la liste finale ? (o): ") != 'o'
    
    return listout

#%% utility functions : get_focus
def get_focus(documents, focus):
    for doc in documents:
        doc['crossTags'] = get_crossTags(doc['raw'], focus) # get focus from raw line

#%% utility functions : get_crossTags
def get_crossTags(raw, focus):
    crossTags=set()
    for key in focus:
        for pattern in focus[key]:
            if pattern.join('  ') in raw : # encadrer pattern de blank pour recherche mot entier
                crossTags.add(key)
    
    return crossTags if len(crossTags)>0 else None

#%% context_dictionay class
class context_dictionary:
    def __init__(self, documents):
        self.kernels = list()
        self.clusters = dict()
        self.scores = dict()
        self.ctxt_dict = dict()
        self.documents = documents
        self.focus = dict()
        self.focus_filters = set()
    
#%%
    def copy(self):
        ctxt_dict_copy = context_dictionary(self.documents)
        ctxt_dict_copy.kernels = self.kernels.copy()
        ctxt_dict_copy.clusters = self.clusters.copy()
        ctxt_dict_copy.scores = self.scores.copy()
        ctxt_dict_copy.ctxt_dict = self.ctxt_dict.copy()
        ctxt_dict_copy.documents = self.documents.copy()
        ctxt_dict_copy.focus = self.focus.copy()
        
        return ctxt_dict_copy
    
#%%
    def update(self, value):
        self.ctxt_dict.update(value)
    
#%%
    def remove_context(self, noun, id):
        self.ctxt_dict[noun].pop(id)
    
#%%
    def remove_thm(self, thm):
        self.ctxt_dict.pop(thm)
        self.kernels.remove(thm)
        self.scores.pop(thm)
    
#%%
    def get_doc(self, idx):
        """Get document by index"""
        found = None
        for doc in self.documents:
            if doc['idx'] == idx:
                found = doc
                break
        
        return found
    
#%%
    def test_focus(self, docIdx):
        """Test si le filtre focus est actif et si le document à l'index docIdx passe le filtre focus
        :return : True si le filtre focus est vide et sinon True si des crossTags existent et s'ils contiennent le filtre focus (entier) en cours"""
        if not self.focus_filters: test = True
        else :
            if self.get_doc(docIdx)['crossTags']: test = self.get_doc(docIdx)['crossTags'].intersection(self.focus_filters) == self.focus_filters
            else : test = False
        
        return test


#%%
    def filter_kernels(self,kernel=""):
        #Filtre les syntagmes en vérifiant que les documents associées sont bien en cohérence avec le filtre du focus en cours
        #Si un kernel est indiqué on ne s'occupe que des syntagmes de ce kernel
        filtered = [(k,[(text,count,[id for id in ids if self.test_focus(id)]) for (text,count,ids) in syntaList]) 
                for (k,syntaList) in self.ctxt_dict.items() if not kernel or k == kernel]
        filtered = [(k,[(text,count,ids) for (text,count,ids) in syntaList if len(ids)>0]) 
                for (k,syntaList) in filtered]
        return [(k,syntaList) for (k,syntaList) in filtered if len(syntaList)>0]
    
#%%
    def print_kernel_syntagms(self, kernel, count=0):
        """print sorted syntagms for given kernel if syntagm count gt count"""
        if self.ctxt_dict[kernel][0][1] >= count: # teste que le premier syntagme est > count
            print('Syntagmes du noyau : %s'%kernel)
            print(com.repeat_str('=',len(kernel)+22))
            idx, nrow = -1, 0
            for ctx in self.ctxt_dict[kernel]:
                n = 0
                if self.focus_filters:
                    printable = False
                    for docIdx in ctx[2]:
                        if self.test_focus(docIdx) : printable=True
                        else : n+=1
                else : printable = True
                
                idx+=1
                if printable and ctx[1]-n > count :
                    print('[%s]: %s (%s)'%(idx,ctx[0],ctx[1]-n))
                    nrow+=1
            print('\n{} syntagmes affichés.\n'.format(nrow))
    
#%%
    def print_cluster_syntagms(self, syntagms, cluster_name, count=1, limit=50):
        """print sorted syntagms if syntagm count gt count"""
        if syntagms[0][1] >= count: # teste que le premier syntagme est > count
            print('Syntagmes du cluster : %s'%cluster_name)
            print(com.repeat_str('=',len(cluster_name)+24))
            idx, nrow = -1, 0
            for idx,ctx in enumerate(syntagms):
                n = 0
                if self.focus_filters:
                    printable = False
                    for docIdx in ctx[2]:
                        if self.test_focus(docIdx) : printable=True
                        else : n+=1
                else : printable = True
                if printable and ctx[1]-n > count and nrow < limit :
                    print('[%s]: %s (%s)'%(idx,ctx[0],ctx[1]-n))
                    nrow+=1
            print('\n{} syntagmes affichés sur {}.'.format(nrow,idx+1))
        ids = 'X'
        while ids !='':
            ids = input('Afficher les verbatims du syntagme [id] ?: ')
            try:
                ids = int(ids)
                title=' '.join(('Verbatims du cluster {',cluster_name,'} pour id syntagme',str(ids)))
                print(title)
                print(com.repeat_str('=',len(title)+1))
                for i in syntagms[ids][2]:
                    if self.test_focus(i): self.print_raw_idx(i)
            except (ValueError, IndexError): print('print_cluster_syntagms ERREUR : Entrée non valide\n')
        
#%%
    def write_context_dict(self, file, count=0):
        """write to file all syntagms for each kernel if syntagm count gt count"""
        with open(file, 'w', encoding='utf-8') as f:
            for noun in self.ctxt_dict:
                if self.ctxt_dict[noun][0][1]>count:
                    f.write('Syntagmes du noyau : %s\n'%noun)
                    f.write(com.repeat_str('=',len(noun)+21)+'\n')
                    for ctx in self.ctxt_dict[noun]:
                        if ctx[1]>count:
                            f.write('%s [%s]\n'%(ctx[0],ctx[1]))
                        else:
                            break
                    f.write('\n')

#%%
    def print_raw_idx(self, idx, wScreen):
        """print raw text from document idx"""
        for doc in self.documents:
            if doc['idx'] == idx :
                print('[{}]: {}'.format(doc['idx'],doc['raw']))
                print(com.repeat_str("-", wScreen))
                break

#%%
    def print_raw_context(self, word, idx=None):
        """print the document raw text including the optional given pattern for a given word"""
        title=' '.join(('Verbatims du noyau',word))
        if idx : title=' '.join((title,'pour id syntagme',str(idx)))
        title+=':'
        print(title)
        print(com.repeat_str('=',len(title)+1))
        if idx :
            for i in self.ctxt_dict[word][idx][2]:
                # affichage du verbatim si test filtre focus True
                if self.test_focus(i): self.print_raw_idx(i)
        else:
            for ctx in self.ctxt_dict[word]:
                for i in ctx[2]:
                    # affichage du verbatim si test filtre focus True
                    if self.test_focus(i): self.print_raw_idx(i)
    
#%%
    def plot_contexts(self, noun, limit=None):
        """Plot histogram de count des contextes du nom donné dans le dictionnaire"""
        if limit: ctxs=self.ctxt_dict[noun][0:limit]
        else: ctxs=self.ctxt_dict['nouns']
        data=[t[1] for t in ctxs]
        
        fig=plt.figure()
        fig.set_size_inches(12,8)
        ax=fig.add_subplot(111)
        
        ind = np.arange(len(data))
        width = 0.35
        
        plt.bar(ind,data,width)
        
        # axes and labels
        ax.set_xlim(-width,len(ind)+width)
        ax.set_ylim(0,max(data)*1.1)
        ax.set_ylabel('Count')
        ax.set_title('Count by context')
        xTickMarks = [t[0] for t in ctxs]
        ax.set_xticks(ind)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=90, fontsize=10)
        
        plt.show()
    
#%%
    def display_focus(self, kernel):
        docIds, documents, counts = set(), list(), list()
        
        # get document list for kernel
        for _,_,docIdxs in self.ctxt_dict[kernel]:
            docIds.update(docIdxs)
        for docId in docIds:
            if self.test_focus(docId): documents.append(self.get_doc(docId))
        
        # count crossTag occurencies
        for tag in self.focus:
            i=0
            for doc in documents:
               if doc['crossTags'] and tag in doc['crossTags']: i+=1
            counts.append(i)
    
        data = pd.DataFrame(counts, index=list(self.focus), columns=['count'])
        
        fig=plt.figure()
        fig.set_size_inches(10,6)
        ax=fig.add_subplot(111)
        ind=np.arange(len(data.index))
        #ind = np.arange(len(data))
        width = 0.45
        
        plt.bar(ind,data['count'],width)
        
        # axes and labels 
        ax.set_xlim(-width,len(ind))
        ax.set_ylim(0,max(data['count'])*1.1)
        ax.set_ylabel('Count')
        ax.set_title('Count by theme')
        xTickMarks = list(data.index)
        ax.set_xticks(ind)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=90, fontsize=10)
        
        plt.show()
    
#%%
    def find_thmKernels(self, thm, model, vocab, limit=None):
        """trouve les noyaux les plus proches du thème donné dans le vocabulaire nettoyé"""
        thms = [ t for t in thm.split() if t in vocab ]
        if thms:
            similar_kernels = [t for t in thms if t in self.kernels ]
            simK_extend = [ k[0] for k in model.most_similar(positive=thms, topn=5000) if k[0] in self.kernels ]
            if limit:
                similar_kernels.extend(simK_extend[0:limit]) #??? filtre extension à 10 noyaux les + similaires ??
            else :
                similar_kernels.extend(simK_extend)
                print("Noyaux similaires :",similar_kernels)
        else:
            print("Warning : aucune similarité sur le thème donné !")
            similar_kernels =  list()
        
        return similar_kernels
        
#%%
    def find_thmSyntagms(self, kernels):
        """trouve les noyaux les plus proches du thème donné"""
        res = list()
        for k in set(kernels):
            res.extend(self.ctxt_dict[k])
        res.sort(key=lambda x: x[1], reverse=True)      
        
        return res
        
#%%
    def thm_syntagms(self, thm, model, vocab):
        """génère les syntagmes d'un thème"""
        similar_kernels = self.find_thmKernels(thm, model, vocab, limit=10)
        res = self.find_thmSyntagms(similar_kernels)
        if res: self.ctxt_dict['__cluster__'] = res
        else : print('Sorry no match, try a list.')
        
#%%
    def thm_focus(self, thm, model, vocab):
        """génère un focus à partir d'un thème"""
        similar_kernels = self.find_thmKernels(thm, model, vocab, limit=25)
        print("Noyaux similaires: "+", ".join(similar_kernels))
        focus = validate_list(similar_kernels)
        if focus:
            print("Focus validé: "+", ".join(focus))
            focus_name = input("Nom du focus ?: ")
            self.focus[focus_name] = focus
            self.focus_filters.add(focus_name)
            print('recalcul des focus...')
            get_focus(self.documents, self.focus)
        else:
            print("Pas de terme dans le focus !")
    
#%%
    def exportJSON(self, filename):
        """export _dic_tabs.json pour analyse dans l'interface shiny"""
        
        # syntagms tab
        dic_tab = list()
        for k in self.ctxt_dict:
            for t in self.ctxt_dict[k]:
                dic_tab.append([k,t[0],t[1],t[3]])
        
        # docidxs tab
        idx_tab = list()
        synt_idx = 1
        for k in self.ctxt_dict:
            for t in self.ctxt_dict[k]:
                for idx in list(t[2]):
                    idx_tab.append([synt_idx,idx])
                synt_idx+=1
        
        # scores tab
        scor_tab = list()
        for k in self.scores:
            scor_tab.append([k,self.scores[k]])
        
        # documents tab
        doc_tab = list()
        for doc in self.documents:
            doc_tab.append([doc['idx'], doc['source'], doc['time'], doc['raw']])
        
        # focus tab
        focus_tab = list()
        for doc in self.documents:
            if 'crossTags' in doc.keys():
                if doc['crossTags']:
                    for ct in doc['crossTags']:
                        focus_tab.append([doc['idx'],ct])
                else:
                    focus_tab.append([doc['idx'],"null"])
        
        # export json
        tabs = {'docs':doc_tab, 'focus':focus_tab, 'synt':dic_tab, 'idxs':idx_tab, 'scor':scor_tab}
        with open(filename+'.json', 'w') as fp:
        	json.dump(tabs, fp)
