import numpy as np
import pandas as pd
import faiss
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Download stopwords list if not already present
nltk.download('stopwords')

class FaissIndexer:
    def __init__(self):
        """
        Initialize the FaissIndexer with a pre-trained embedding model.
        """
        self.index = None
        self.original_data = None  # To store original data for reference
        self.stop_words = set(stopwords.words('english'))  # Set of stop words for cleaning text
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained embedding model

    def transform_vector_column(self, df, text_column):
        """
        Transform and preprocess text data in a DataFrame column using a pre-trained embedding model.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        - text_column (str): Name of the column containing text data.

        Returns:
        - np.ndarray: Numpy array of vectors.
        """
        try:
            # Convert text data to embeddings
            vectors = df[text_column].apply(lambda x: self.embedding_model.encode(str(x)).astype('float32'))
            valid_vectors = np.vstack(vectors)
            self.original_data = df.reset_index(drop=True)  # Keep track of the original data
            print(f"Transformed and preprocessed {len(valid_vectors)} valid vectors.")
            return valid_vectors
        except Exception as e:
            print(f"Failed to preprocess vector data: {e}")
            return None

    def build_index(self, data):
        try:
            dimension = data.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(data)
            print(f"FAISS index built with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"Failed to build index: {e}")

    def save_index(self, file_path):
        try:
            faiss.write_index(self.index, file_path)
            print(f"Index saved to {file_path}.")
        except Exception as e:
            print(f"Failed to save index: {e}")

    def load_index(self, file_path):
        try:
            self.index = faiss.read_index(file_path)
            print(f"Index loaded from {file_path}.")
        except Exception as e:
            print(f"Failed to load index: {e}")

    def clean_text(self, text):
        """
        Clean and preprocess input text by removing stop words and special characters.

        Parameters:
        - text (str): The input text.

        Returns:
        - str: Cleaned text.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        words = text.split()
        cleaned_text = ' '.join([word for word in words if word not in self.stop_words])
        return cleaned_text

    def text_to_vector(self, text):
        """
        Convert cleaned text to a vector representation using a pre-trained embedding model.

        Parameters:
        - text (str): The input text.

        Returns:
        - np.ndarray: Vector representation of the text.
        """
        cleaned_text = self.clean_text(text)
        try:
            vector = self.embedding_model.encode(cleaned_text).astype('float32')
            return vector
        except Exception as e:
            print(f"Failed to convert text to vector: {e}")
            return None

    def search(self, query_vector, k=5, neighbor_sensitivity=50):
        """
        Search for the nearest neighbors of the given query vector.

        Parameters:
        - query_vector (np.ndarray): The query vector to search with.
        - k (int): Default number of nearest neighbors to retrieve.
        - neighbor_sensitivity (int): User-defined sensitivity value from 0 to 100.

        Returns:
        - list: List of tuples containing distances and original data of the nearest neighbors.
        """
        try:
            # Adjust `k` based on user-defined sensitivity (scales from 1 to k based on sensitivity)
            adjusted_k = max(1, int(k * (neighbor_sensitivity / 100)))
            query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
            
            if self.index is None:
                print("Error: FAISS index is not built or loaded.")
                return None
            if query_vector.shape[1] != self.index.d:
                print(f"Error: Dimension mismatch. Query vector has {query_vector.shape[1]} dimensions, expected {self.index.d}.")
                return None
            
            distances, indices = self.index.search(query_vector, adjusted_k)
            neighbors = [
                (dist, self.original_data.iloc[idx]) for dist, idx in zip(distances[0], indices[0])
            ]
            return neighbors
        except Exception as e:
            print(f"Search failed: {e}")
            return None

    def search_by_text(self, text, k=5, neighbor_sensitivity=50):
        query_vector = self.text_to_vector(text)
        if query_vector is None:
            print("Error: Failed to generate a query vector from the input text.")
            return None
        return self.search(query_vector, k, neighbor_sensitivity)

