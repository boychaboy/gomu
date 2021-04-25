import os
import csv
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from metric import read_data

def top_n_words(n, data, output_file):
    with open(fw, 'w') as fw:
         


        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="path to input file") 
    args = parser.parse_args()

    df_data = read_data(args.input_file)
    counts = dict()
    for index, data in df_data.iterrows():
        sent1 = data['sent1']
        sent2 = data['sent2']
        sent1 = sent1.strip().split()
        sent2 = sent2.strip().split()
        word_set = set()
        
        for word1, word2 in zip(sent1, sent2):
            word_set.add(word1)
            word_set.add(word2)
        for word in word_set:
            counts[word] = counts.get(word,0)+1

    keywords = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sw = stopwords.words('english')

    import ipdb; ipdb.set_trace(context=10)
    # print(counts)
    print(f"Number of words : {len(keywords)}")
    print()
    print(f"Top 30 words : {keywords[:30]}")
     
    keywords_non_stopwords = []
    cnt = 0
    # remove stopwords
    for word in keywords:
        if word[0].lower() in sw:
            continue
        keywords_non_stopwords.append(word)
        cnt += 1 
        if cnt > 30:
            break
    print()
    print(f"Top 30 words (not stopwords) : {keywords_non_stopwords}")
    



if __name__ == "__main__":
    print("gomu")
    main()


