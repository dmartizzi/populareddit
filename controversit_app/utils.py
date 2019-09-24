import os.path
from os import path
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
import gensim
import gensim.corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import common_corpus
from gensim.test.utils import datapath
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import sklearn
import sklearn.model_selection as skmodsel
import sklearn.linear_model as sklinmod
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pickle

def get_list_of_subr(nmf_model_path):
    '''
    Reads list of available NMF models from 
    the appropriate path and creates a list 
    of subreddits for which the data is 
    available.
    '''
    list_bash_command = "ls "+nmf_model_path
    process = subprocess.Popen(list_bash_command.split(), stdout=subprocess.PIPE)
    list_of_nfm_models, error = process.communicate()
    list_of_subr = list_of_nfm_models.decode("utf-8").split()

    headlen = len("nmf_model_prod_r-")
    for i in range(0,len(list_of_subr)):
        list_of_subr[i]=str(list_of_subr[i])[headlen:]

    return list_of_subr


def load_nmf_model(nmf_model_path,subr):
    outpath = nmf_model_path
    nmfoutfile = outpath+"/nmf_model_prod_r-"+subr
    if path.exists(nmfoutfile):
        nmf_model = pickle.load(open(nmfoutfile, 'rb'))
        success_nmf = True
    else:
        nmf_model = "NaN"
        success_nmf = False
        
    vecoutfile = outpath+"/count_vectorizer_prod_r-"+subr
    if path.exists(nmfoutfile):
        vectorizer = pickle.load(open(vecoutfile, 'rb'))
        success_vec = True
    else:
        vectorizer = "NaN"
        success_vec = False

    traoutfile = outpath+"/tfidf_transformer_prod_r-"+subr
    if path.exists(traoutfile):
        transformer = pickle.load(open(traoutfile, 'rb'))
        success_tra = True
    else:
        transformer = "NaN"
        success_tra = False
        
    return nmf_model,success_nmf,vectorizer,success_vec,transformer,success_tra

def load_reg_model(reg_model_path,subr):
    outpath = reg_model_path
    
    regoutfile = outpath+"/gbreg_popular_prod_r-"+subr
    if path.exists(regoutfile):
        #print(regoutfile+"model exists!")
        reg_pop_model = pickle.load(open(regoutfile, 'rb'))
        success_pop = True
    else:
        reg_pop_model = "NaN"
        success_pop = False

    regoutfile = outpath+"/gbreg_controversial_prod_r-"+subr
    if path.exists(regoutfile):
        #print(regoutfile+"model exists!")
        reg_con_model = pickle.load(open(regoutfile, 'rb'))
        success_con = True
    else:
        reg_con_model = "NaN"
        success_con = False
    
    regoutfile = outpath+"/gbprecision_nmf_prod_r-"+subr+".csv"
    if path.exists(regoutfile):
        sc = pd.read_csv(regoutfile)
        score_val_pop = float(sc["PopValidationScore"][0])*100
        score_val_con = float(sc["ConValidationScore"][0])*100
        pop_fraction = "%.1f"%float(sc["PopFraction"][0]*100)
        con_fraction = "%.1f"%float(sc["ConFraction"][0]*100)
    else :
        score_val_pop = 0.0
        score_val_con = 0.0
        pop_fraction = "0.0"
        con_fraction = "0.0"
        
    return reg_pop_model,score_val_pop,success_pop,reg_con_model,score_val_con,success_con,pop_fraction,con_fraction

def lemmatize_stemming(text):
    '''Function to lemmatize text'''
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    '''Function to pre-process text'''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def run_full_model(subm_text,nmf_model,vectorizer,transformer,reg_pop_model,reg_con_model):
    # Pre-process string
    processed_string = preprocess(subm_text)
    test_sentence = [' '.join(text) for text in processed_string]
    x_counts = vectorizer.transform(test_sentence)
    x_tfidf = transformer.transform(x_counts)
    x_tfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

    # Get topic array
    y = nmf_model.transform(x_tfidf_norm)
    predicted_topic_nmf = normalize(y, norm='l1', axis=1)

    # Predict popularity and controversiality
    popular = reg_pop_model.predict(predicted_topic_nmf)
    controversial = reg_con_model.predict(predicted_topic_nmf)

    return popular, controversial

