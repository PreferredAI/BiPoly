import os

import networkx as nx
import pandas as pd
from networkx.classes.graph import Graph


def recursive_add_edge(
    G: Graph, vertex_name: str, root_name: str, delimiter: str
):
    if delimiter not in vertex_name:
        G.add_edge(root_name, vertex_name)
        return

    split_name = vertex_name.rsplit(delimiter, 1)
    parent = split_name[0].strip()
    child = vertex_name.strip()
    G.add_edge(parent, child)

    if delimiter in vertex_name:
        recursive_add_edge(G, parent, root_name, delimiter)


def convert_nodes_to_int(
    G: Graph,
    filename: str,
    write_dot: bool = True,
    output_path: str = None,
    first_label: int = 0,
    old_label: str = "old_label",
    del_old: bool = True,
):
    print("Converting labels...")
    G2 = nx.convert_node_labels_to_integers(
        G, label_attribute=old_label, first_label=first_label
    )
    print("Converting labels complete.")

    if not output_path:
        output_path = os.path.join(os.getcwd(), "output/")

    print("Writing legend...")
    lengend = nx.get_node_attributes(G2, old_label)
    pd.DataFrame(lengend, index=["Name"]).T.to_csv(
        os.path.join(output_path, filename + ".csv")
    )
    print("Writing legend complete.")

    if del_old:
        for node in G2:
            del G2.nodes[node][old_label]

    if write_dot:
        print("Writing dot...")
        nx.nx_pydot.write_dot(G2, os.path.join(output_path, filename + ".dot"))
        print("Writing dot complete.")

    return G2
