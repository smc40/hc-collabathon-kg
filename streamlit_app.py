import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
from src.utils import DATA_FILE

st.set_page_config(layout="wide")

with open(DATA_FILE) as dfile:
    df = pd.read_csv(dfile)


st.title('Bob Blobs')


nodes = []
edges = []
nodes.append( Node(id="Spiderman", 
                   label="Peter Parker", 
                   size=25)
            ) # includes **kwargs
nodes.append( Node(id="Captain_Marvel", 
                   size=25) 
            )
edges.append( Edge(source="Captain_Marvel", 
                   label="friend_of", 
                   target="Spiderman", 
                   # **kwargs
                   ) 
            ) 

config = Config(width=750,
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
