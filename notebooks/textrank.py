import pke
import pandas as pd
import regex_extraction

pos = {'NOUN', 'PROPN', 'ADJ'}
extractor = pke.unsupervised.TextRank()

def getCandidatePhrases(transcript):
    key_pos = {}
    transcript = [regex_extraction.cleantext(transcript)]
    for seg in transcript:
        extractor.load_document(input=seg, language='en', normalization=None)
        extractor.candidate_weighting(window=2, pos=pos,top_percent=0.33)
        keyphrases = extractor.get_n_best(n=1000)
        df = pd.DataFrame({
            "Keyphrase":[i for i,j in keyphrases],
            "POS":[j for i,j in keyphrases]
        })
    return df
