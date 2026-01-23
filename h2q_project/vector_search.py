import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define constants for embedding model and data file paths
EMBEDDING_MODEL = 'all-mpnet-base-v2'
DATA_DIR = 'data'
INDEX_FILE = os.path.join(DATA_DIR, 'index.json')

class VectorSearch:
    def __init__(self, model_name: str = EMBEDDING_MODEL, index_file: str = INDEX_FILE):
        self.model = SentenceTransformer(model_name)
        self.index_file = index_file
        self.index = self.load_index()

    def load_index(self) -> Dict:
        """Loads the pre-computed index from a JSON file."""
        if not os.path.exists(self.index_file):
            print(f"Index file not found: {self.index_file}")
            return {}
        with open(self.index_file, 'r') as f:
            return json.load(f)

    def build_index(self, data: List[Dict], text_key: str) -> None:
        """Builds the index from a list of dictionaries.

        Args:
            data: A list of dictionaries, where each dictionary contains the text to be indexed.
            text_key: The key in each dictionary that contains the text.
        """
        self.index = {}
        for i, item in enumerate(data):
            text = item.get(text_key, '')
            if text:
                embedding = self.model.encode(text)
                self.index[str(i)] = {
                    'embedding': embedding.tolist(),
                    'metadata': item
                }
        self.save_index()

    def save_index(self) -> None:
        """Saves the index to a JSON file."""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(self.index_file, 'w') as f:
            json.dump(self.index, self.index, indent=4)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Searches the index for the given query.

        Args:
            query: The query string.
            top_k: The number of results to return.

        Returns:
            A list of dictionaries, where each dictionary contains the metadata of the matching document.
        """
        query_embedding = self.model.encode(query)
        results = []
        for key, value in self.index.items():
            embedding = np.array(value['embedding'])
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
            results.append((key, similarity, value['metadata']))

        results.sort(key=lambda x: x[1], reverse=True)
        return [result[2] for result in results[:top_k]]

    def add_document(self, document: Dict, text_key: str) -> None:
        """Adds a single document to the index.

        Args:
            document: The document to add.
            text_key: The key in the document that contains the text.
        """
        text = document.get(text_key, '')
        if not text: return

        new_index_id = str(len(self.index))
        embedding = self.model.encode(text)
        self.index[new_index_id] = {
            'embedding': embedding.tolist(),
            'metadata': document
        }
        self.save_index()


    def delete_document(self, index_id: str) -> None:
        """Deletes a document from the index.

        Args:
            index_id: The ID of the document to delete.
        """
        if index_id in self.index:
            del self.index[index_id]
            self.save_index()
        else:
            print(f"Document with index ID {index_id} not found.")


    def update_document(self, index_id: str, document: Dict, text_key: str) -> None:
        """Updates a document in the index.

        Args:
            index_id: The ID of the document to update.
            document: The updated document.
            text_key: The key in the document that contains the text.
        """
        if index_id in self.index:
            text = document.get(text_key, '')
            if not text: return

            embedding = self.model.encode(text)
            self.index[index_id] = {
                'embedding': embedding.tolist(),
                'metadata': document
            }
            self.save_index()
        else:
            print(f"Document with index ID {index_id} not found.")