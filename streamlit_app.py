import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
from src.utils import DATA_FILE, WEIGHT_FILE

st.set_page_config(layout="wide")

with open(DATA_FILE) as dfile:
    df_nodes = pd.read_csv(dfile)

df_nodes.dropna(subset=["id_trial"], inplace=True)

with open(WEIGHT_FILE) as dfile:
    df_edges = pd.read_csv(dfile)

#extract unique cuids
node_names = list(set(df_nodes["id_trial"]))
unique_edge_nodes = set(df_edges['pm_ref']).union(set(df_edges['pm_rel']))
node_names = [n for n in node_names if n in unique_edge_nodes]

nodes = []
edges = []

for n in unique_edge_nodes:
    nodes.append(Node(
        id=n,
        label = n,
        size=20
    ))

st.title('Bob Blobs')

for index, (source, target, weight) in df_edges.iterrows():
    print(source + " " + target)
    edges.append(
        Edge(
            source=source,
            target=target
        )
    )


config = Config(width=1250,
                height=950,
                directed=False, 
                physics=True, 
                hierarchical=False,
                # **kwargs
                )

return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)




<<<<<<< HEAD
=======


>>>>>>> 0321921 (Initial graph)
