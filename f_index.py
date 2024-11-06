import numpy as np
import pandas as pd
import faiss
import re

class FaissIndexer:
    def __init__(self):
        """
        Initialize the FaissIndexer.
        """
        self.index = None
        self.original_data = None  # To store original data for reference

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
    def transform_vector_column(self, df, vector_column):
        """
        Transform and preprocess vector data in a DataFrame column.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        - vector_column (str): Name of the column containing vector data.

        Returns:
        - np.ndarray: Numpy array of vectors.
        """
        try:
            # Function to clean and convert string representation of vectors
            def clean_and_convert(vector_string):
                try:
                    if not isinstance(vector_string, str) or vector_string.strip() == "":
                        return None
                    # Extract only numbers from the string
                    vector_string = re.sub('[^0-9.,-]', '', vector_string.strip('[]'))
                    vector = np.fromstring(vector_string, sep=',')
                    if vector.size > 0 and not np.isnan(vector).any():
                        return vector
                    else:
                        return None
                except Exception as ex:
                    print(f"Failed to convert vector: {vector_string}. Error: {ex}")
                    return None
                
            # Apply the cleaning and transformation function
            df['vector'] = df[vector_column].apply(clean_and_convert)
            # Filter out rows with invalid or None vectors
            valid_data = df[df['vector'].notna()]
            valid_vectors = [v for v in valid_data['vector'].tolist() if v is not None and len(v) > 0]

            if len(valid_vectors) == 0:
                raise ValueError("No valid vectors found after preprocessing.")

            dimension = len(valid_vectors[0])
            valid_vectors = [v for v in valid_vectors if len(v) == dimension]

            # Store original data for reference
            self.original_data = valid_data.reset_index(drop=True)
            # Convert list of vectors to a 2D numpy array
            data = np.vstack(valid_vectors).astype('float32')
            print(f"Transformed and preprocessed {len(data)} valid vectors.")
            return data
        except Exception as e:
            print(f"Failed to preprocess vector data: {e}")
            return None

    def build_index(self, data):
        """
        Build a FAISS index from the data.

        Parameters:
        - data (np.ndarray): Numpy array of data to index.
        """
        try:
            dimension = data.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(data)
            print(f"FAISS index built with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"Failed to build index: {e}")

    def save_index(self, file_path):
        """
        Save the FAISS index to a file.

        Parameters:
        - file_path (str): Path to save the index file.
        """
        try:
            faiss.write_index(self.index, file_path)
            print(f"Index saved to {file_path}.")
        except Exception as e:
            print(f"Failed to save index: {e}")

    def load_index(self, file_path):
        """
        Load a FAISS index from a file.

        Parameters:
        - file_path (str): Path of the index file to load.
        """
        try:
            self.index = faiss.read_index(file_path)
            print(f"Index loaded from {file_path}.")
        except Exception as e:
            print(f"Failed to load index: {e}")

    def search(self, query_vector, k=5):
        """
        Search the index for nearest neighbors and reference original data.

        Parameters:
        - query_vector (np.ndarray): The query vector.
        - k (int): Number of nearest neighbors to find.

        Returns:
        - (list): Distances and original data entries of nearest neighbors.
        """
        try:
            query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
            distances, indices = self.index.search(query_vector, k)
            neighbors = [
                (dist, self.original_data.iloc[idx]) for dist, idx in zip(distances[0], indices[0])
            ]
            return neighbors
        except Exception as e:
            print(f"Search failed: {e}")
            return None

    def text_to_vector(self, text):
        """
        Convert text to a vector representation.

        Parameters:
        - text (str): The input text.

        Returns:
        - np.ndarray: Vector representation of the text.
        """
        # Placeholder function for converting text to a vector
        # Replace this with your actual text-to-vector conversion logic (e.g., using embeddings)
        vector = np.random.rand(self.index.d).astype('float32')  # Example random vector
        return vector

    def search_by_text(self, text, k=5):
        """
        Search for the closest neighbors to the given text.

        Parameters:
        - text (str): The input text to search for.
        - k (int): Number of nearest neighbors to find.

        Returns:
        - (list): Distances and original data entries of nearest neighbors.
        """
        query_vector = self.text_to_vector(text)
        return self.search(query_vector, k)

if __name__ == "__main__":
    # Initialize the indexer
    indexer = FaissIndexer()

    # Path to your CSV file
    csv_file_path = 'data_sample.csv'

    # Load data into a DataFrame
    df = indexer.load_data_from_csv(csv_file_path)

    if df is not None:
        # Name of the column containing vector data
        vector_column_name = 'idphrase'

        # Transform and preprocess vector data
        data = indexer.transform_vector_column(df, vector_column_name)

        if data is not None:
            # Build the FAISS index
            indexer.build_index(data)

            # Optionally, save the index to a file
            index_file_path = 'faiss_index.index'
            indexer.save_index(index_file_path)

            # Load the index from the file (if needed)
            # indexer.load_index(index_file_path)

            # Perform a text-based search
            search_text = "cancer"
            neighbors = indexer.search_by_text(search_text, k=5)

            print(f"\nSearch results for text query: {search_text}")
            for dist, original_data in neighbors:
                print(f"Distance: {dist}, Original Data: {original_data}")
