from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import log
import numpy
from collections import Counter
from whoosh.analysis import SimpleAnalyzer, StopFilter

dir_path = "C:/Users/amrmr/OneDrive/Desktop/data"

class Document:
    def __init__(self, doc_num): 
        text = open(f"{dir_path}/document_{doc_num}.txt", "r", encoding='utf-8').read()
        self.dim = set(Document.tokenize_line(text))
        self.tf = dict()
        self.tf_calculator(text)
        self.vector = dict()
        self.reduced_vector = dict()

    def tf_calculator(self, text):
        term_counter = Counter()
        tokenized_text = Document.tokenize_line(text)
        for term in self.dim:
            term_counter[term] = tokenized_text.count(term)
        for term in self.dim:
            self.tf[term] = term_counter[term] / len(tokenized_text)

    @staticmethod
    def tokenize_line(txt):
        analyzer = SimpleAnalyzer() | StopFilter()
        tokens = [token.text for token in analyzer(txt)]
        return tokens

    def doc_vector_cal(self, idf):
        for term in idf:
            self.vector[term] = self.tf[term] * idf[term]



class DocumentCluster:
    def __init__(self, doc_list):
        self.doc_dict = dict()
        self.top_terms = set()  # Set to store the top 10 terms

        for doc_num in doc_list:
            doc = Document(doc_num)  # Use an empty set for dimensions
            self.doc_dict[doc_num] = doc

            # Update top_terms with the top 10 terms from the current document
            self.top_terms.update(self.find_top_terms(doc, 10))

        # Calculate document-level vectors
        self.doc_level_vector_calculator()

        # Reduce dimensionality using PCA
        self.reduce_dimensions()

        # Cluster documents
        self.cluster_documents()

    def find_top_terms(self, document, n):
        # Get the top n terms with the highest TF scores in the document
        sorted_terms = sorted(document.tf.items(), key=lambda x: x[1], reverse=True)
        top_terms = set(term for term, _ in sorted_terms[:n])
        return top_terms

    def doc_level_vector_calculator(self):
        term_counter = Counter()
        for doc in self.doc_dict:
            for term in self.top_terms:  # Use top_terms instead of all_terms
                if term not in self.doc_dict[doc].tf:
                    self.doc_dict[doc].tf[term] = 0
                elif self.doc_dict[doc].tf[term] != 0:
                    term_counter[term] += 1
        idf = dict()
        for term in self.top_terms:  # Use top_terms instead of all_terms
            idf[term] = log(len(self.doc_dict) / (term_counter[term] + 1))

        # Calculate document-level vectors
        for doc in self.doc_dict:
            self.doc_dict[doc].doc_vector_cal(idf)

    def reduce_dimensions(self):
        # Extract document vectors using only the top terms
        doc_vectors = [list(doc.vector.values()) for doc in self.doc_dict.values()]

        # Use PCA to reduce dimensions
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(doc_vectors)

        # Add reduced vectors back to the Document objects
        for i, doc_num in enumerate(self.doc_dict):
            self.doc_dict[doc_num].reduced_vector = reduced_vectors[i]

    def cluster_documents(self):
        # Extract reduced vectors for clustering
        vectors_for_clustering = [doc.reduced_vector for doc in self.doc_dict.values()]

        # Use KMeans for clustering
        kmeans = KMeans(n_clusters=3, n_init=10)  # You can adjust the number of clusters as needed
        clusters = kmeans.fit_predict(vectors_for_clustering)

        # Plot the clusters
        self.plot_clusters(vectors_for_clustering, clusters)

    def plot_clusters(self, vectors, labels):
        vectors_array = numpy.array(vectors)  # Convert the list of vectors to a NumPy array
        plt.scatter(vectors_array[:, 0], vectors_array[:, 1], c=labels, cmap='viridis')
        plt.title('Document Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

if __name__ == "__main__":
    Doc_list = input("Enter the document numbers: ").split(" ")
    cluster = DocumentCluster(Doc_list)