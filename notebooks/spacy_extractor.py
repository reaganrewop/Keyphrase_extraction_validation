import pandas as pd
import numpy as np
import spacy
import pickle
nlp = spacy.load('en_core_web_sm')
stop_words = list(nlp.Defaults.stop_words)

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import regex_extraction

def getCandidatePhrases(transcript):
    key_pos = {}
    transcript = [regex_extraction.cleantext(transcript)]
    for seg in transcript:
        for sent in sent_tokenize(seg):
            key = nlp(sent)
            for each_key in list(key.noun_chunks):
                key_pos[each_key] = list(token.pos_ for token in nlp(str(each_key)))
    df = pd.DataFrame({
        "Keyphrase":[' '.join([str(i) for i in key]) for key in list(key_pos.keys())],
        "POS":list(key_pos.values())
    })
    return df
