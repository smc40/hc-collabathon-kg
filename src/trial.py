import src.utils as utl
from streamlit_agraph import agraph, Node, Edge, Config

def _get_default_graph_data():
    df_edges = utl.load_edges()

    cluster_nodes = ["NCT05029583", "NCT05583344", "NCT01186952", "NCT05844644"]
    df_edges = df_edges[(df_edges["pm_ref"].isin(cluster_nodes)) | (df_edges["pm_rel"].isin(cluster_nodes))]

    # Extract unique cuids
    unique_edge_nodes = set(df_edges['pm_ref']).union(set(df_edges['pm_rel']))

    nodes = []
    edges = []

    for n in unique_edge_nodes:
        nodes.append(Node(
            id=n,
            label = n,
            size=20
        ))

    for index, (source, target, weight) in df_edges.iterrows():
        edges.append(
            Edge(
                source=source,
                target=target
            )
        )

    return nodes, edges


def _get_serch_graph_data():
    search_query = 'Denosumab'
    neighbors = utl.search_index(search_query)
    cluster_nodes = [n['id_trial'] for _, n in neighbors]

    df_edges = utl.load_edges()
    df_edges = df_edges[(df_edges["pm_ref"].isin(cluster_nodes)) | (df_edges["pm_rel"].isin(cluster_nodes))]

    # Extract unique cuids
    unique_edge_nodes = set(df_edges['pm_ref']).union(set(df_edges['pm_rel']))

    nodes = []
    edges = []

    for n in unique_edge_nodes:
        nodes.append(Node(
            id=n,
            label = n,
            size=20
        ))

    for index, (source, target, weight) in df_edges.iterrows():
        edges.append(
            Edge(
                source=source,
                target=target
            )
        )

    return nodes, edges

if __name__ == '__main__':
    default_nodes, default_edges = _get_default_graph_data()
    searched_nodes, searched_edges = _get_serch_graph_data()

    foo = 1

    # nodes = ['NCT01935102', 'NCT01935102', 'NCT01935102']
    # df_nodes = utl.load_nodes()
    # nodes = [n['id_trial'] for _, n in df_nodes.iterrows()]
    # edges = utl.load_edges(nodes=nodes)
    #
    # search_query = 'Denosumab'
    # neighbors = utl.search_index(search_query)
    #
    # foo = 1

