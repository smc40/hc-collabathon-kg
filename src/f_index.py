import numpy as np
import pandas as pd
import faiss
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import os
import torch
from tqdm import tqdm  # Import tqdm for progress bar

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

        # Load a pre-trained embedding model and move it to GPU if available
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for SentenceTransformer.")
            self.embedding_model = self.embedding_model.to('cuda')
        else:
            print("CUDA is not available. Using CPU for SentenceTransformer.")

    def load_data_from_csv(self, file_path):
        """
        Load data from a CSV file into a DataFrame.

        Parameters:
        - file_path (str): Path to the CSV file.

        Returns:
        - pd.DataFrame: DataFrame containing data loaded.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded DataFrame with {len(df)} rows from {file_path}.")
            return df
        except Exception as e:
            print(f"Failed to load data from CSV: {e}")
            return None

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
            print("Transforming text data to vectors...")
            # Use tqdm to show progress
            vectors = df[text_column].apply(lambda x: self.embedding_model.encode(
                str(x),
                convert_to_numpy=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ).astype('float32'), raw=False)
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

            # Use GPU if available for FAISS
            if faiss.get_num_gpus() > 0:
                print("CUDA is available for FAISS. Using GPU.")
                res = faiss.StandardGpuResources()  # Create a GPU resource object
                self.index = faiss.IndexFlatL2(dimension)  # Build the CPU index first
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # Move index to GPU
            else:
                print("CUDA is not available for FAISS. Using CPU.")
                self.index = faiss.IndexFlatL2(dimension)

            self.index.add(data)
            print(f"FAISS index built with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"Failed to build index: {e}")

    def save_index(self, file_path):
        try:
            faiss.write_index(faiss.index_gpu_to_cpu(self.index) if faiss.get_num_gpus() > 0 else self.index, file_path)
            print(f"Index saved to {file_path}.")
        except Exception as e:
            print(f"Failed to save index: {e}")

    def load_index(self, file_path):
        try:
            self.index = faiss.read_index(file_path)
            if faiss.get_num_gpus() > 0:
                print("Loading index to GPU for FAISS.")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
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
        text = re.sub(r'[^a-zA-Z\\s]', '', text)  # Remove special characters
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
            vector = self.embedding_model.encode(
                cleaned_text,
                convert_to_numpy=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ).astype('float32')
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
        if self.original_data is None:
            print("Error: Original data is not available for reference.")
            return None

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

    def process_csv_and_build_or_load_index(self, csv_file_path, text_column):
        """
        Process a CSV file and either build a new index or load an existing one.

        Parameters:
        - csv_file_path (str): Path to the CSV file.
        - text_column (str): Name of the column containing text data.
        """
        index_file_path = f"{os.path.splitext(csv_file_path)[0]}_faiss_index.index"
        synced_data_file_path = f"{os.path.splitext(csv_file_path)[0]}_synced_data.csv"

        if os.path.exists(index_file_path) and os.path.exists(synced_data_file_path):
            print(f"Index file {index_file_path} and data file {synced_data_file_path} exist. Loading index and data...")
            self.load_index(index_file_path)
            self.original_data = self.load_data_from_csv(synced_data_file_path)
        else:
            print(f"Index or synced data file not found. Building new index...")
            df = self.load_data_from_csv(csv_file_path)
            if df is not None:
                data = self.transform_vector_column(df, text_column)
                if data is not None:
                    self.build_index(data)
                    self.save_index(index_file_path)
                    # Save the DataFrame to ensure it is in sync with the index
                    df.to_csv(synced_data_file_path, index=False)
                    print(f"Synced data saved to {synced_data_file_path}.")
