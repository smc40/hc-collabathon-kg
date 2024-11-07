import streamlit as st
from networkx.algorithms.traversal import dfs_edges
from streamlit_agraph import agraph, Node, Edge, Config

import src.utils as utl

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

if search_query:
    st.text(f'You are searching for: {search_query}')
    neighbors = utl.search_index(search_query)

    node_names = [n["id_trial"] for _, n in neighbors]
    df_edges = utl.load_edges(nodes=node_names)
    node_names = list(set(df_edges['pm_ref'].values).union(set(df_edges['pm_rel'].values)))

    nodes = []
    for nde in node_names:
        nodes.append(Node(
            id=nde,
            label=nde,
            size=20
        ))

    edges = []
    for _, (source, target, weight) in df_edges.iterrows():
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
                    )

    return_value = agraph(nodes=nodes, edges=edges, config=config)
