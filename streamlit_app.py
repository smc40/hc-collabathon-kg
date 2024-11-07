import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd

import src.utils as utl

def _get_edges(search_query: str | None = None) -> pd.DataFrame:
    if search_query:
        neighbors = utl.search_index(search_query)
        nodes = [n["id_trial"] for _, n in neighbors]
        df_edges = utl.load_edges(nodes=nodes)
    else:
        df_edges = utl.load_edges()

    return df_edges

st.set_page_config(page_title='Bobs Blobs',
                   page_icon=":spider_web:",
                   layout='wide')

st.markdown("""
    <style>
    .title {
        text-align: center;
        padding-bottom: 40px;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #dfe1e5;
    } 
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='title'>Clinical Assessment Reports Network</h1>", unsafe_allow_html=True)

_, srch, _ = st.columns([1, 3, 1])
with srch:
    search_query = st.text_input("Search", placeholder="Search Titles", label_visibility="collapsed")

df_edges = _get_edges(search_query)

min_val = min(0.0, df_edges['weight_cosine'].min())
max_val = max(1.0, df_edges['weight_cosine'].max())
threshold = st.slider('Weight Threshold', min_value=min_val, max_value=max_val, value=min_val, step=0.01)
df_edges = df_edges[df_edges['weight_cosine'] >= threshold]

if search_query:
    st.text(f'You are searching for: {search_query}')


node_names = list(set(df_edges['pm_ref'].values).union(set(df_edges['pm_rel'].values)))
node_name_title_map = {row['pm_ref']: row['phrase_ref'] for _, row in df_edges.iterrows()}
node_name_title_map.update({row['pm_rel']: row['phrase_rel'] for _, row in df_edges.iterrows()})

node_id_template = '{name}: {title}'
nodes = []
for name in node_names:
    title = node_name_title_map[name]
    nodes.append(Node(
        # id=node_name_title_map[name],
        id=node_id_template.format(name=name, title=title),
        label=name,
        size=20
    ))
edges = []
for _, (src_name, tgt_name, _, weight, tgt_title, src_title) in df_edges.iterrows():
    edges.append(
        Edge(
            source=node_id_template.format(name=src_name, title=src_title),
            target=node_id_template.format(name=tgt_name, title=tgt_title),
        )
    )
config = Config(width=2160,
                height=950,
                directed=False,
                physics=True,
                hierarchical=False,
                )
return_value = agraph(nodes=nodes, edges=edges, config=config)
