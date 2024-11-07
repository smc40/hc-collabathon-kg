import src.utils as utl

if __name__ == '__main__':
    nodes = ['NCT01935102', 'NCT01935102', 'NCT01935102']
    df_nodes = utl.load_nodes()
    nodes = [n['id_trial'] for _, n in df_nodes.iterrows()]
    edges = utl.load_edges(nodes=nodes)
