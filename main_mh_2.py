
# Data Structure Project
# Search Engine(Phase 2)
# Collabrators: Seyyed Amirmohammad Mirshamsi, Mohammadhossein Damad

from whoosh.analysis import SimpleAnalyzer, StopFilter
import os
from collections import Counter
from math import log
'''
from sklearn.cluster import KMeans

X = [[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
'''
'''
import matplotlib.pyplot as plt

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.show()
'''
class Static:
    @staticmethod
    def tokenize(text):
        analyzer = SimpleAnalyzer() | StopFilter()
        tokens = [token.text for token in analyzer(text)]
        return tokens
    
    docs_tokenized_words = list()

class Document:
    def __init__(self, doc_num):
        self.doc_vector_dims = set(Static.docs_tokenized_words[doc_num])
        self.doc_dims_tfs = self.doc_dims_tfs_calculator(doc_num)
        self.doc_dims_idfs = self.doc_dims_idfs_calculator()
        self.doc_vector = self.doc_vector_builder()

    def doc_dims_tfs_calculator(self, doc_num):
        term_counter = Counter()
        for dim in self.doc_vector_dims:
            term_counter[dim] = 0
        for term in Static.docs_tokenized_words[doc_num]:
            term_counter[term] += 1
        dims_tfs = dict()
        for dim in term_counter:
            dims_tfs[dim] = term_counter[dim] / len(Static.docs_tokenized_words[doc_num])
        return dims_tfs

    def doc_dims_idfs_calculator(self):
        doc_counter = Counter()
        for dim in self.doc_vector_dims:
            doc_counter[dim] = 0
        for dim in self.doc_vector_dims:
            for doc_words in Static.docs_tokenized_words:
                if dim in doc_words:
                    doc_counter[dim] += 1
        dims_idfs = dict()
        for dim in doc_counter:
            dims_idfs[dim] = log(6 / (1 + doc_counter[dim]))
        return dims_idfs
    
    def doc_vector_builder(self):
        tf_idf = list()
        for term in self.doc_vector_dims:
            tf_idf.append(self.doc_dims_tfs[term] * self.doc_dims_idfs[term])
        return tf_idf

class Program:
    def __init__(self):
        self.docs_vectors_list = list()
        index = 0
        while(index != 6):
            Static.docs_tokenized_words.append(Static.tokenize(open(os.getcwd() + "\\data\\document_" + str(index) + ".txt", "r", encoding='utf-8').read()))
            index += 1
        index = 0
        while(index != 6):
            self.docs_vectors_list.append(Document(index).doc_vector)
            index += 1

if __name__ == "__main__":
    system = Program()