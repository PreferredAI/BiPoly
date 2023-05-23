import os
from typing import Collection

import config
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from entity.candidate import Candidate

from module import graphing


def draw_bipartite(
    candidate_pair: Collection[Candidate],
    raw_solution,
    func: callable = None,
    prefix: str = "matched",
):
    if func:
        suffix = "-" + func.__name__
        args = "-".join(("{}={}".format(*i) for i in func.keywords.items()))
        if args:
            suffix += "-" + args
    else:
        suffix = ""

    offset = len(candidate_pair[0].G)

    G = nx.DiGraph()
    G.add_nodes_from([str(i) for i in range(offset)], bipartite=0)
    G.add_nodes_from(
        [str(i) for i in range(offset, offset + len(candidate_pair[1].G))],
        bipartite=1,
    )

    G.add_edges_from(
        [(str(x), str(y + offset)) for x, y in raw_solution["x_route"]]
    )
    G.add_edges_from(
        [(str(y + offset), str(x)) for y, x in raw_solution["y_route"]]
    )

    pos = nx.bipartite_layout(G, [str(i) for i in range(offset)])
    nx.draw(G, pos=pos, with_labels=True)
    plt.savefig(
        os.path.join(config.OUTPUT_DIR, "{}-{}.pdf".format(prefix, suffix))
    )


def draw(
    candidate_pair: Collection[Candidate],
    raw_solution,
    func: callable = None,
    prefix: str = "matched",
):

    if func:
        suffix = "-" + func.__name__
        args = "-".join(("{}={}".format(*i) for i in func.keywords.items()))
        if args:
            suffix += "-" + args
    else:
        suffix = ""

    callgraph = pydot.Dot(graph_type="digraph", compound="true")

    offset = 0
    name = None
    for candidate in candidate_pair:
        G = candidate.G
        if name and name == candidate.name:
            name = candidate.name + "_"
        else:
            name = candidate.name
        G_i = graphing.convert_nodes_to_int(
            G,
            "matched-{}{}".format(name, suffix),
            old_label="tooltip",
            del_old=False,
            first_label=offset,
        )
        offset += len(G)
        G_dot = nx.nx_pydot.to_pydot(G_i)

        cluster = pydot.Cluster(name, label=name)
        callgraph.add_subgraph(cluster)
        for node in G_dot.get_nodes():
            cluster.add_node(node)
        for edge in G_dot.get_edges():
            cluster.add_edge(edge)

    x_nodes = list(candidate_pair[0].G)
    y_nodes = list(candidate_pair[1].G)

    offset -= len(G)
    for x, y in raw_solution["x_route"]:
        callgraph.add_edge(
            pydot.Edge(
                str(x),
                str(y + offset),
                color="red",
                tooltip="{} -> {}".format(x_nodes[x], y_nodes[y]),
            )
        )

    for y, x in raw_solution["y_route"]:
        callgraph.add_edge(
            pydot.Edge(
                str(y + offset),
                str(x),
                color="red",
                tooltip="{} -> {}".format(y_nodes[y], x_nodes[x]),
            )
        )

    print("Writing raw dot...")
    callgraph.write_dot(
        os.path.join(config.OUTPUT_DIR, "{}-raw{}.dot".format(prefix, suffix))
    )
    print("Writing raw dot complete.")

    print("Drawing graph...")
    callgraph.write_svg(
        os.path.join(config.OUTPUT_DIR, "{}-{}.svg".format(prefix, suffix))
    )
    print("Drawing graph complete.")

    print(
        "Type{} objective found: {}".format(suffix, raw_solution["objective"])
    )


def draw_circular(
    candidate_pair: Collection[Candidate],
    raw_solution,
    func: callable = None,
    prefix: str = "matched",
):

    if func:
        suffix = "-" + func.__name__
        args = "-".join(("{}={}".format(*i) for i in func.keywords.items()))
        if args:
            suffix += "-" + args
    else:
        suffix = ""

    offset = len(candidate_pair[0].G)

    G = nx.Graph()
    G.add_nodes_from([str(i) for i in range(offset)], bipartite=0)
    G.add_nodes_from(
        [str(i) for i in range(offset, offset + len(candidate_pair[1].G))],
        bipartite=1,
    )

    G.add_edges_from(
        [(str(x), str(y + offset)) for x, y in raw_solution["x_route"]]
    )
    G.add_edges_from(
        [(str(y + offset), str(x)) for y, x in raw_solution["y_route"]]
    )

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos)
    # nx.draw(G, pos=pos)
    plt.savefig(
        os.path.join(config.OUTPUT_DIR, "{}-{}.pdf".format(prefix, suffix))
    )
