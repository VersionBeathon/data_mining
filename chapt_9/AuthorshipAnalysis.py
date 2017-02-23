#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/23 9:36
# @Author :Kira
# @Software：PyCharm
import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import grid_search
from sklearn.feature_extraction.text import CountVectorizer


data_folder = r"C:\Users\Administrator\Data\books"

def clean_book(document):
    lines = document.split("\n")
    start = 0
    end = len(lines)
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("*** START OF THIS PROJECT GUTENBERG"):
            start = i+1
        elif line.startswith("*** END OF THIS PROJECT GUTENBERG"):
            end = i - 1
    return "\n".join(lines[start:end])


def load_books_data(folder=data_folder):
    documents = []
    authors = []
    subfolders = [subfolder for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    for author_number, subfolder in enumerate(subfolders):
        full_subfloder_path = os.path.join(folder, subfolder)
        for document_name in os.listdir(full_subfloder_path):
            with open(os.path.join(full_subfloder_path, document_name), encoding="utf-8") as inf:
                documents.append(clean_book(inf.read()))
                authors.append(author_number)
    return documents, np.array(authors, dtype='int')
documents, classes = load_books_data(data_folder)

function_words = [  "a", "able", "aboard", "about", "above", "absent",
                    "according" , "accordingly", "across", "after", "against",
                    "ahead", "albeit", "all", "along", "alongside", "although",
                    "am", "amid", "amidst", "among", "amongst", "amount", "an",
                    "and", "another", "anti", "any", "anybody", "anyone",
                    "anything", "are", "around", "as", "aside", "astraddle",
                    "astride", "at", "away", "bar", "barring", "be", "because",
                    "been", "before", "behind", "being", "below", "beneath",
                    "beside", "besides", "better", "between", "beyond", "bit",
                    "both", "but", "by", "can", "certain", "circa", "close",
                    "concerning", "consequently", "considering", "could",
                    "couple", "dare", "deal", "despite", "down", "due", "during",
                    "each", "eight", "eighth", "either", "enough", "every",
                    "everybody", "everyone", "everything", "except", "excepting",
                    "excluding", "failing", "few", "fewer", "fifth", "first",
                    "five", "following", "for", "four", "fourth", "from", "front",
                    "given", "good", "great", "had", "half", "have", "he",
                    "heaps", "hence", "her", "hers", "herself", "him", "himself",
                    "his", "however", "i", "if", "in", "including", "inside",
                    "instead", "into", "is", "it", "its", "itself", "keeping",
                    "lack", "less", "like", "little", "loads", "lots", "majority",
                    "many", "masses", "may", "me", "might", "mine", "minority",
                    "minus", "more", "most", "much", "must", "my", "myself",
                    "near", "need", "neither", "nevertheless", "next", "nine",
                    "ninth", "no", "nobody", "none", "nor", "nothing",
                    "notwithstanding", "number", "numbers", "of", "off", "on",
                    "once", "one", "onto", "opposite", "or", "other", "ought",
                    "our", "ours", "ourselves", "out", "outside", "over", "part",
                    "past", "pending", "per", "pertaining", "place", "plenty",
                    "plethora", "plus", "quantities", "quantity", "quarter",
                    "regarding", "remainder", "respecting", "rest", "round",
                    "save", "saving", "second", "seven", "seventh", "several",
                    "shall", "she", "should", "similar", "since", "six", "sixth",
                    "so", "some", "somebody", "someone", "something", "spite",
                    "such", "ten", "tenth", "than", "thanks", "that", "the",
                    "their", "theirs", "them", "themselves", "then", "thence",
                    "therefore", "these", "they", "third", "this", "those",
                    "though", "three", "through", "throughout", "thru", "thus",
                    "till", "time", "to", "tons", "top", "toward", "towards",
                    "two", "under", "underneath", "unless", "unlike", "until",
                    "unto", "up", "upon", "us", "used", "various", "versus",
                    "via", "view", "wanting", "was", "we", "were", "what",
                    "whatever", "when", "whenever", "where", "whereas",
                    "wherever", "whether", "which", "whichever", "while",
                    "whilst", "who", "whoever", "whole", "whom", "whomever",
                    "whose", "will", "with", "within", "without", "would", "yet",
                    "you", "your", "yours", "yourself", "yourselves"]

extractor = CountVectorizer(vocabulary=function_words)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = SVC()
grid = grid_search.GridSearchCV(svr, parameters)
pipeline1 = Pipeline([('feature_extraction', extractor), ('clf', grid)])
pipeline = Pipeline([('feature_extraction', CountVectorizer(analyzer='char', ngram_range=(3, 3))),
                    ('classifier', grid)])
