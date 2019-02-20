# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python [conda env:DL-wpython3]
#     language: python
#     name: conda-env-DL-wpython3-py
# ---

'''
import xmltodict

with open('/home/ray/Programming/testing/ake-datasets/datasets/ACM/test/299468.xml') as fd:
    doc = xmltodict.parse(fd.read())

File = open(' ')
    #for sent in range(doc['root']['document']['sentences']['sentence']['@id']):
    #    print ()
    #    break
    #print(doc['root']['document']['sentences']['sentence'][3]['tokens']['token'][0]['word'])
    #print(doc['root']['document']['sentences']['sentence'][3]['tokens']['token'][0]['POS'])
print(len(doc['root']['document']['sentences']['sentence']))
print()
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import glob
import json
import pandas as pd
from nltk.stem.snowball import SnowballStemmer as Stemmer
import regex_extraction
import spacy_extractor
import textrank
import nltk
from nltk.corpus import stopwords
import numpy as np
stop_words = set(stopwords.words('english'))

inputfiles = glob.glob("../data/raw/all_docs_abstacts_refined/*.txt")
inputfiles = [files.replace(".txt","") for files in inputfiles]
references_key = {}
references_text = {}
#inputfiles = ['1005058']
custom_regex = []
spacy = []
textrank_key = []
tp_all=[0,0,0]
fp_all=[0,0,0]
fn_all=[0,0,0]
file_count =0
for single_file in inputfiles:
        with open(single_file + '.key', 'r') as f:
                lines = f.readlines()
                keyphrases = [line.rstrip().lower() for line in lines]
                references_key[single_file] = keyphrases
        with open( single_file + '.txt', 'r') as f:
                lines = f.read()
                references_text[single_file] = lines
        custom_regex = regex_extraction.getCandidatePhrases(references_text[single_file])
        spacy = spacy_extractor.getCandidatePhrases(references_text[single_file])
        textrank_key = textrank.getCandidatePhrases(references_text[single_file])

        true_key = references_key[single_file]
        true_key = [' '.join([k for k in key.split(' ') if k not in stop_words]) for key in true_key ]
        true_key_dict = {}
        for key in true_key:
                true_key_dict[key] = 0
        for i,algo in enumerate([custom_regex,spacy,textrank_key]):
                check_key = list(algo['Keyphrase'])
                check_key = [' '.join([k for k in key.split(' ') if k not in stop_words]) for key in check_key ]
                tp=0
                fp=0
                fn=0
                for key in check_key:
                        if len(key.split(' '))>1:
                                if key in true_key:
                                        tp+=1
                                        if true_key_dict[key] ==1:
                                                tp-=1
                                        else:
                                                true_key_dict[key]=1
                                elif any(key in word for word in true_key):
                                        tp+=0.5
                                else:
                                        fp+=1
                for key in true_key_dict.keys():
                        if true_key_dict[key] == 0:
                                fn+=1
                tp_all[i]+=tp
                fp_all[i]+=fp
                fn_all[i]+=fn
        file_count+=1
        if (file_count==10):
            break
for i,algo in enumerate([custom_regex,spacy,textrank_key]):
        recall = tp_all[i]/(tp_all[i]+fn_all[i])
        print("recall for  " + {0:"custom_Regex",1:"spacy",2:"textrank"}[i] + " is: " + str(recall))


