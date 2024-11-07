from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.f_index import FaissIndexer

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data'

NODES_FILE = DATA_DIR / 'data_sample_subgraph.csv'
EDGES_FILE = DATA_DIR / 'full-graph-weights-cosine.csv'

# _CLUSTER_NODES = ["NCT05029583", "NCT05583344", "NCT01186952", "NCT05844644"]


def load_nodes() -> pd.DataFrame:
    with open(NODES_FILE) as dfile:
        df = pd.read_csv(dfile)
    df = df.dropna(subset=['id_trial'])
    return df

def load_edges(nodes: List | None = None) -> pd.DataFrame:
    with open(EDGES_FILE) as dfile:
        df = pd.read_csv(dfile)

    if nodes:
        df = df[(df["pm_ref"].isin(nodes)) | (df["pm_rel"].isin(nodes))]
    return df

_INDEXER = FaissIndexer()
df = load_nodes()
data = _INDEXER.transform_vector_column(df, 'idphrase')
_INDEXER.build_index(data)
_INDEXER.save_index('faiss_index.index')

def search_index(text: str, k: int=5, neighbor_sensitivity: int=75) -> List[Tuple[float, pd.Series]]:
    neighbors = _INDEXER.search_by_text(text, k=k, neighbor_sensitivity=neighbor_sensitivity)
    return neighbors

if __name__ == "__main__":
    search_text = "cancer"
    neighbors = search_index(search_text)

    if neighbors:
        print(f"\nSearch results for text query: {search_text}")
        for dist, original_data in neighbors:
            print(f"Distance: {dist}, Original Data: {original_data}")
