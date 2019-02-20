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

import time
import numpy as np
import pandas as pd
from re import finditer
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import nltk
import itertools

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

punct = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))

def lambda_unpack(f):
    return lambda args: f(*args)

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"We'll": "We will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


def cleantext(text):
    rep = {"\n": " ", "\t": " ", "--": " ", "--R": " ", ";": " ","(":" ",")":" ","[":" ","]":" ",",":" ","#":" "}
    substrs = sorted(rep, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    text =  regexp.sub(lambda match: rep[match.group(0)], text)

    text = replaceContractions(text)
    return text

def replaceContractions(text):
    c_filt_text = ''
    for word in word_tokenize(text):
        if word in contractions:
            c_filt_text = c_filt_text+' '+contractions[word]
        else:
            c_filt_text = c_filt_text+' '+word
    return c_filt_text


def extract_candidate_chunk(text_all, grammar=r'KT: {<(CD)|(DT)|(JJR)>*( (<NN>+ <NN.>+)|((<JJ>|<NN>) <NN>)| ((<JJ>|<NN>)+|((<JJ>|<NN>)* (<NN> <NN.>)? (<JJ>|<NN>)*) <NN.>)) <VB.>*}'):
    chunker = nltk.RegexpParser(grammar)
    candidates_all = []
    key_pos = []
    for text in sent_tokenize(text_all):
        if text!=" " and text!="":
            #print (text,[word_tokenize (sent) for sent in sent_tokenize (text)])
            tagged_sents = nltk.pos_tag ([word_tokenize (sent) for sent in sent_tokenize (text)] [0])
            all_chunks = itertools.chain.from_iterable([nltk.chunk.tree2conlltags(chunker.parse(tagged_sents)) for tagged_sent in tagged_sents])
            candidates = [' '.join(word for word,pos, chunk in group).lower() for key,group in itertools.groupby(all_chunks, lambda_unpack(lambda word,pos,chunk: chunk !='O')) if key]
            candidates_all += candidates
    valid_key = list(set([cand for cand in candidates_all if cand not in stop_words and not all(char in punct for char in cand)]))
    for key in valid_key:
        key_pos.append([x[1] for x in nltk.pos_tag([key][0].split(' '))])
    return valid_key,key_pos


def extract_candidate_words(text_all, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    candidate_all = []
    key_pos = []
    for text in sent_tokenize(text_all):
        if text!='' and text!=' ':
            tagged_words = nltk.pos_tag([word_tokenize(sent) for sent in sent_tokenize(text)][0])
            candidates = [word.lower() for word, tag in tagged_words
                          if tag in good_tags and word.lower() not in stop_words
                          and not all(char in punct for char in word)]
            candidate_all += candidates
    for key in candidate_all:
        key_pos.append([x[1] for x in nltk.pos_tag([key][0].split(' '))])
    return candidate_all,key_pos

'''
def getCandidatePhrases(transcript):
    input_ = replaceContractions(transcript)
    Keywords_all = list (set (extract_candidate_chunk (transcript) + extract_candidate_words (transcript)))
    return Keywords_all
'''

def getCandidatePhrases(transcript):
    key_pos = {}
    transcript = [cleantext(transcript)]
    for seg in transcript:
        chunk_key,chunk_pos = extract_candidate_chunk (seg)
        word_key,word_pos = extract_candidate_words (seg)
        key_all = chunk_key + word_key
        pos_all = chunk_pos + word_pos
        for i in range(len(key_all)):
            key_pos[key_all[i]] = pos_all[i]
    df = pd.DataFrame({
        "Keyphrase":list(key_pos.keys()),
        "POS":list(key_pos.values())
    })
    return df

getCandidatePhrases("With a foundation in artificial intelligence and media analytics, Ether starts its course by enabling a smart call service on top of Slack, Stride, and Teams. Ether captures and analyzes the call (audio, video, shared content, etc) as the call happens and extracts key markers.")
