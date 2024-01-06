
# Data Structure Project
# Search Engine(Phase 2)
# Collabrators: Seyyed Amirmohammad Mirshamsi, Mohammadhossein Damad

from whoosh.analysis import SimpleAnalyzer, StopFilter
import os
from collections import Counter
from math import log
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Static:
    @staticmethod
    def tokenize(text):
        analyzer = SimpleAnalyzer() | StopFilter()
        tokens = [token.text for token in analyzer(text)]
        return tokens

    @staticmethod
    def reducedimensions(vectors_list):
        return PCA(n_components=2).fit_transform(vectors_list)
    
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
        while(index != 1001):
            Static.docs_tokenized_words.append(Static.tokenize(open(os.getcwd() + "\\data\\document_" + str(index) + ".txt", "r", encoding='utf-8').read()))
            index += 1
        index = 0
        while(index != 1001):
            self.docs_vectors_list.append(Document(index).doc_vector)
            index += 1
        temp = list()
        index = 0
        while(index != 1001):
            i = 5
            while(i != 0):
                temp.append(max(self.docs_vectors_list[index]))
                i -= 1
            self.docs_vectors_list[index].clear()
            for item in temp:
                self.docs_vectors_list[index].append(item)
            temp.clear()
            index += 1
        self.docs_reduced_dims_vectors_list = Static.reducedimensions(self.docs_vectors_list)
        self.kmeans = KMeans(n_clusters = 3, random_state = 0).fit(self.docs_reduced_dims_vectors_list)
        plt.scatter(self.docs_reduced_dims_vectors_list[:, 0], self.docs_reduced_dims_vectors_list[:, 1], c=self.kmeans.labels_, cmap='rainbow')
        plt.show()

if __name__ == "__main__":
    system = Program()
