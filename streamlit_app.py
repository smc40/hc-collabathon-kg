import streamlit as st
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

    nodes = [n["id_trial"] for _, n in neighbors]
    print(nodes)

    edges = utl.load_edges(nodes=nodes)
    print(len(edges))

    for nde in nodes:
        nodes.append(Node(
            id=nde,
            label=nde,
            size=20
        ))


    for _, (source, target, weight) in edges.iterrows():
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

    agraph(nodes=nodes, edges=edges, config=config)
