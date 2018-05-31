# -*- coding: utf-8 -*-

#%% LIB IMPORTS
import os
import sys
import time

import re #import the regular expressions library; will be used to strip punctuation
from collections import Counter #allows for counting the number of occurences in a list
import pandas as pd

import treetaggerwrapper as tagr

import nltk_vars as vars

TREE_TAGGER_RDIR = ''

#%% VARS
tag_abbreviations ={'ABR' : 'Abreviation (TV)',\
                    'ADJ' : 'Adjectif',\
                    'ADV' : 'Adverbe',\
                    'DET:ART' : 'Article',\
                    'DET:POS' : 'Pronom Possessif (ma, ta, ...)',\
                    'INT' : 'Interjection',\
                    'KON' : 'Conjunction',\
                    'NAM' : 'Nom Propre',\
                    'NOM' : 'Nom',\
                    'NUM' : 'Numéral',\
                    'PRO' : 'Pronom',\
                    'PRO:DEM' : 'Pronom Démonstratif',\
                    'PRO:IND' : 'Pronom Indefini',\
                    'PRO:PER' : 'Pronom Personnel',\
                    'PRO:POS' : 'Pronom Possessif (mien, tien, ...)',\
                    'PRO:REL' : 'Pronom Relatif',\
                    'PRP' : 'Préposition',\
                    'PRP:det' : 'Préposition + Article (au,du,aux,des)',\
                    'PUN' : 'Ponctuation',\
                    'PUN:cit' : 'Ponctuation de citation',\
                    'SENT' : 'Balise de phrase',\
                    'SYM' : 'Symbole',\
                    'VER:cond' : 'Verbe au conditionnel',\
                    'VER:futu' : 'Verbe au futur',\
                    'VER:impe' : 'Verbe à l’impératif',\
                    'VER:impf' : 'Verbe à l’imparfait',\
                    'VER:infi' : 'Verbe à infinitif',\
                    'VER:pper' : 'Verbe au participe passé',\
                    'VER:ppre' : 'Verbe au participe présent',\
                    'VER:pres' : 'Verbe au présent',\
                    'VER:simp' : 'Verbe au passé simple',\
                    'VER:subi' : 'Verbe à l’imparfait du subjunctif',\
                    'VER:subp' : 'Verbe au présent du subjunctif'}

#%% NLTK FUNCTION DEFINITIONS

# Il faut remplacer le None par "C:/Program Files/Java/jdk1.8.0_131/bin" dans le param bin "C:\Applications\Anaconda3\lib\site-packages\nltk\internals.py", line 58, in config_java

root_path = "." #define a working directory path
os.chdir(root_path) #set the working directory path

#%%
def preprocessing_raw(raw):
    """cleaning and preprocessing raw text"""
    text = re.sub(r'[\n]$', '', raw) #filter out eol carriage return chars
    text = re.sub('&amp;','&', text) #convert out html cars
    text = re.sub('^"','',text) # filter out double quotes in begining of string
    text = re.sub('"$','',text) # filter out double quotes in end of string
    text = re.sub('r[\x00-\x1f\x7f-\x9f]','',text) # filter out non printable characters
    return text

#%%
def clean_text(raw):
    """turn the raw text into an clean nltk text object"""
    text = raw.lower() # lowering
    text = re.sub('http[s]?:[/][/][a-z0-9_./]*', '_url_', text) # masquage des urls
    text = re.sub('[a-z0-9_]*(/[a-z0-9_]+)+.[a-z0-9]{3,4}', '_file_', text) # masque files
    text = re.sub('[a-z0-9_\.]*[@#][a-z0-9_]*[\.a-z]*','_@_',text) # filter mail, hashtags & at
#    text = re.sub(r'\s[+-]?[0-9]*[\.]?[0-9]+[^gs]\s',' _num_ ',text) # mask nums
    text = re.sub(r'[\n!;,?:\.()]', ' . ', text) # conversion ponctuation de fin de phrase
    text = re.sub('[,/%$€@°#<>\^\t]',' ',text) # remplace les caractères inutiles
    text = re.sub(r'\\n', ' . ', text) # filter out this strange pattern
    text = "".join([c for c in text if c.isprintable()]) # filter out non printable chars
    # monograms
    for m in vars.monograms:
        text = re.sub('\\s'+m[0]+'(\')?\\s',' '+m[1]+' ',text)
    text = re.sub("([cdjnmlst])\'",r"\1e ",text)
    text = re.sub(r'[\'\`’\"«»]',' ',text) # corrige les apostrophes
    text = re.sub('[&()<>,\[\]]','',text) # filter out cars
    text = re.sub('[\s]{2,}', ' ', text) #replace multiple spaces by one
    return text.strip()

#%%
def filter_stopwords(words,stopwords):
    """normalizes the words by turning them all lowercase and then filters out the stopwords"""
    #filtering stopwords
    filtered_words = [ word for word in words if (word not in stopwords and word.isprintable() and len(word) > 1) ]
    filtered_words.sort() #sort filtered_words list
    return filtered_words

#%%
def sort_dictionary(dictionary):
    """returns a sorted dictionary (as tuples) based on the value of each key"""
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

#%%
def normalize_counts(counts):
    total = sum(counts.values())
    return dict((word, float(count)/total) for word,count in counts.items())
        
#%%
def print_sorted_dictionary(tuple_list, limit=None):
    """print the results of sort_dictionary"""
    i=0
    for tup in tuple_list:
        print('{:.3f}'.format(tup[1]) + '\t' + tup[0])
        i+=1
        if limit and i==limit: break

#%%
def list_tagtype(corpus, tagtype):
    """list all unique tokens of corpus from the specified tag type"""
    res=set()
    for doc in corpus:
        for tag,ttag in doc['tags']:
            if ttag==tagtype.upper():
                res.add(tag)
    print(res)

#%%
def print_pos_tags(tags):
    """print all the tags with their part of speech; tag[0] is the word; tag[1] is the Part of Speech"""
    for tag in tags: print('\t'.join(tag))

#%%
def get_pos_tags(tags, poss=None):
    """gets all the tags with their part of speech; tag.word is the word; tag.pos is the Part of Speech and tag.lemma is the lemma form.
    :param tags: list of tags
    :param pos: list of pos token
    :return: list of tokens
    :rtype: list"""

    get_tags = list()
    
    for pos in poss :
        tag_abbreviations_upper = {k.upper():v for k,v in tag_abbreviations.items()}
        if pos.upper() in tag_abbreviations_upper:
            for tag in tags: 
                if tag.pos.upper()==pos.upper(): get_tags.append(tag.lemma)
        else:
            print("%s is not a valid search term." %(pos))
    return get_tags

#%%
def list_pos_tags(corpus, pos=None):
    pos_tags=set()
    for doc in corpus:
        for tag in doc['tags']:
            if tag[1]==pos.upper(): pos_tags.add(tag[0])
    return pos_tags

#%%
def search_pos(tags,search_term,pos):
    """looks for a particular POS word prior to the search term, see what comes after the search term"""
    print("POS\tPREC\t\tS.TERM\t\tSUC\n")
    for i,tag in enumerate(tags):
        if tags[i-1][1].upper()==pos.upper() and tag[0].lower()==search_term.lower():
            print(str(i)+'\t'+tags[i-1][0]+"\t" + tag[0] + "\t" + tags[i+1][0])

#%%
def get_doc(documents, idx):
    """Renvoie un document par son index
    :param documents: document list au format dictionnaire
    :param idx: index du document à renvoyer"""
    
    for d in documents:
        if d['idx'] == idx:
            return d

#%%
def remove_duplicates(documents):
    unique_sens = set()
    unique_doc_ids = list()
    for d in documents:
        sen = " ".join([ t.word for t in d['tags'] ])
        if sen not in unique_sens:
            unique_doc_ids.append(d['idx'])
            unique_sens.add(sen)
    docs = [ get_doc(documents, docIdx) for docIdx in unique_doc_ids ]
    
    return docs

#%% extrait les documents du corpus
def get_documents(docs, stopwords):
    """Extrait les documents du corpus
    :param corpus: [(source,datetime,text)]"""
    
    documents=list()
    corpus = [ (doc[0],doc[1],doc[2]) for doc in docs if len(doc[2].split()) > 3 ] # supprime les lignes courtes de moins de 4 mots
    
    tagger = tagr.TreeTagger(TAGLANG='fr', TAGDIR='c:/Applications/TreeTagger', TAGPARFILE='C:/Applications/TreeTagger/lib/french-utf8.par')
    
    idx, start_time = 1, time.time()
    for doc in corpus:
        source, datetime, raw = doc[0], doc[1], doc[2]
        tags = tagr.make_tags(tagger.tag_text(clean_text(raw)))
        tags = [ tag for tag in tags if type(tag) == tagr.Tag ]
        # add all our elements to the array (documents)
        # each element in the array is a dictionary
        documents.append({ 'idx':idx, 'source':source, 'time':datetime, 'raw':raw, 'tags':tags })
        idx = progress_per(idx, len(corpus), start_time) # print the progress percentage info
    print()
    
    return documents

#%%
def repeat_str(pattern,times):
    res=''
    for i in range(1,times): res+=pattern
    return res

#%%
def print_doc(doc):
    """print info about a single document"""
    line='\n'+repeat_str('=',len(doc['file']+str(doc['idx']))+5)+'\n'
    print(doc['file'],': ',doc['idx'],line,doc['raw'])
    for tag in doc['tags']:
        print(''.join((tag[0],' (',tag_abbreviations[tag[1]],') ')), end='')

#%%
def print_raw(documents, pattern=None):
    """print out the raw text from documents wich includes the search pattern"""
    if pattern:
        for doc in documents:
            if pattern in doc['tokens']: print_doc(doc)
    else:
        for doc in documents: print_doc(doc)

#%% progress bar
def progress_per(i, total, start_time):
    """print a progress string in percent of i/total"""
    c = b'\xe2\x96\x88'.decode('utf-8')
    bar_length = 50 # en nb characters
    p = int(round(i/total*bar_length,0))
    bar = c*p+' '*(bar_length-p)
    estimated_time = (total/i)*(time.time()-start_time)
    if estimated_time < 60 : estimated_time = "~{:.0f} sec.".format(estimated_time)
    elif estimated_time < 3600 : estimated_time = "~{:.0f} min.".format(estimated_time/60)
    else : estimated_time = "~{:.0f} h.".format(estimated_time/3600)
    sys.stdout.write("{:.1f}% |{}| {}/{} {}\r".format(round(i/total*100,1), bar, i, total, estimated_time))
    sys.stdout.flush()
    return i+1

#%% top nouns
def get_top_nouns(documents, stopwords=None):
    all_nouns = [ noun for doc in documents for noun,_ in doc['nouns'] ]
    if stopwords: top_nouns=sort_dictionary(Counter(filter_stopwords(all_nouns,stopwords)))
    else: top_nouns=sort_dictionary(Counter(all_nouns))
    return top_nouns    

#%% get_phrases
def get_phrases(tags):
    """retourne les phrases de la liste de tags si un nom est présent
    :return: list of sentences or None il no sentence fits the rule"""
    sentences, phrase = list(), list()
    t = tags.copy()
    t.append(tagr.Tag(word='SENT', pos='SENT', lemma='SENT'))
    for tag in t:
        if tag.pos != 'SENT':
            phrase.append(tag)
        else :
            poss = [tag.pos for tag in phrase]
            if any(pos in {'NOM','NAM'} for pos in poss) and len(poss) > 1:
                sentences.append(phrase)
            phrase = list()

    return(sentences if len(sentences) >0 else None)
    
#%% crossTags functions
def get_crossTags_tx(documents, crossThemes):
    tags,counts=list(),list()
    for tag,_  in crossThemes:
        tags.append(tag)
        i=0
        for doc in documents:
           if doc['crossTags'] and tag in doc['crossTags']: i+=1
        counts.append(i)
    return pd.DataFrame(counts, index=tags, columns=['count'])

def print_crossTags_tx(documents, crossThemes):
    for tag,_  in crossThemes:
        i=0
        for doc in documents:
           if doc['crossTags'] and tag in doc['crossTags']: i+=1
        print('Taux de docs croisés %s : %.2f%% [%s]'%(tag,i/len(documents)*100,i))

#%%
def docfilter_crossTags(documents, crossTags):
    """filter documents by the given cross tag
    
    :param documents: corpus to filter - a list of document
    :param crossTags: cross tag list
    :return: list of filtered document (new corpus)"""

    filteredDocs=list()
    idxs_to_keep=set()
    idx=0
    # constitution set des doc_idxs to remain
    for doc in documents:
        for tag in crossTags:
            if doc['crossTags'] and tag in doc['crossTags']: idxs_to_keep.add(idx)
        idx+=1
    # filtered docs list with doc_index to keep
    for idx in idxs_to_keep:
        filteredDocs.append(documents[idx])
    # indexing new docs
    idx=0
    for doc in filteredDocs:
        doc['idx'] = idx
        idx+=1
        
    return filteredDocs
