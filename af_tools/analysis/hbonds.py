import subprocess
import tempfile
import networkx as nx
import re

import seaborn as sns
import matplotlib.pyplot as plt
import collections
from natsort import natsorted


def find_hbonds(models: list[str]):
    # TODO: finding H bonds without ChimeraX ?

    with tempfile.NamedTemporaryFile(suffix=".cxc") as f:
        for model in models:
            f.write(f"open {model};\n".encode())
            f.write(
                f"hbonds #1/* restrict #1/* log true intraMol false interModel false;\n"
                .encode())
            f.write(
                "hbonds #1/* restrict #1/* log true intraMol false interModel false saltonly true;\n"
                .encode())
            f.write("close #1;\n\n".encode())
        f.flush()

        cmd = ["chimerax", "--exit", "--nogui", f.name]
        p = subprocess.run(cmd, capture_output=True, text=True)

    s = p.stdout.split("Chain information for")
    graphs = [nx.Graph() for _ in range(len(models))]

    for i, model_s in enumerate(s[1:]):
        l = model_s.split(
            "H-bonds (donor, acceptor, hydrogen, D..A dist, D-H..A dist):")

        bond_ex = r".+[A-Z]{3}.+[A-Z]{3}.+\d{3}"
        hbonds = re.findall(bond_ex, l[1])
        saltbridges = re.findall(bond_ex, l[2])

        bond_group_ex = r"\/([A-z]+) ([A-Z]{3}) (\d+)"
        for hbond_str in hbonds:
            matches = list(re.finditer(bond_group_ex, hbond_str))

            # chain id must be different
            assert matches[0].group(1) != matches[1].group(1)

            v1 = f"{matches[0].group(1)}-{matches[0].group(3)}-{matches[0].group(2).title()}"
            v2 = f"{matches[1].group(1)}-{matches[1].group(3)}-{matches[1].group(2).title()}"

            graphs[i].add_node(v1)
            graphs[i].add_node(v2)

            if (v1, v2) not in graphs[i].edges:
                graphs[i].add_edge(v1, v2, count=1, saltbridge_count=0)
            else:
                graphs[i].edges[(v1, v2)]["count"] += 1

        for saltbrige_str in saltbridges:
            matches = list(re.finditer(bond_group_ex, saltbrige_str))

            # chain id must be different
            assert matches[0].group(1) != matches[1].group(1)

            v1 = f"{matches[0].group(1)}-{matches[0].group(3)}-{matches[0].group(2).title()}"
            v2 = f"{matches[1].group(1)}-{matches[1].group(3)}-{matches[1].group(2).title()}"

            graphs[i].edges[(v1, v2)]["saltbridge_count"] += 1
    return graphs


def draw_hbonds_graph(G: nx.classes.graph.Graph,
                      model_count: int,
                      figsize: tuple[float, float] = (12, 12),
                      salt_color: tuple[float, float,
                                        float] = sns.color_palette()[9],
                      edge_color: tuple[float, float, float] = (0, 0, 0),
                      red_color: tuple[float, float,
                                       float] = sns.color_palette()[3],
                      red_treshold: float = 0.15,
                      min_edge_alpha: float = 0.1,
                      max_edge_alpha: float = 1,
                      node_color: dict | str = "w",
                      node_edge_color: dict | str = "#2C404C"):
    from netgraph import Graph
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(facecolor='red', figsize=(12, 12))

    layers = collections.defaultdict(list)
    node_labels = {}
    for n in natsorted(G.nodes, reverse=True):
        s = n.split("-")
        layers[s[0]].append(n)
        node_labels[n] = f"{s[1]}{s[2]}"
    layers = list(dict(sorted(layers.items())).values())  # type: ignore

    edge_labels = nx.get_edge_attributes(G, "count")
    red_color = sns.color_palette()[3]

    edge_colors = {}
    edge_alphas = {}
    max_edge_count = max(nx.get_edge_attributes(G, "count").values())
    for e in G.edges:
        edge_alphas[e] = min_edge_alpha + (max_edge_alpha - min_edge_alpha) * (
            G.edges[e]["count"] / max_edge_count)

        if G.edges[e]["count"] < red_treshold * model_count:
            edge_colors[e] = red_color
            edge_labels[e] = None
            continue

        if G.edges[e]["saltbridge_count"] != 0:
            edge_colors[e] = salt_color
            edge_labels[
                e] = f"{G.edges[e]['count']} [{G.edges[e]['saltbridge_count']} / {G.edges[e]['count']} ] ({G.edges[e]['count'] / model_count:.3f})"
        else:
            edge_colors[e] = edge_color
            edge_labels[
                e] = f"{G.edges[e]['count']} ({G.edges[e]['count'] / model_count:.3f})"

    graph = Graph(G,
                  node_layout="multipartite",
                  node_color=node_color,
                  node_edge_color=node_edge_color,
                  node_size=1,
                  edge_layout="straight",
                  edge_label_rotate=False,
                  edge_labels=edge_labels,
                  edge_color=edge_colors,
                  edge_alpha=edge_alphas,
                  node_labels=node_labels,
                  node_label_offset=(0.00, 0),
                  node_layout_kwargs=dict(layers=layers))
    for n in G.nodes:
        graph.node_label_artists[n].set_size("medium")
        pos = graph.node_label_artists[n].get_position()
        if n.startswith("A"):
            graph.node_label_artists[n].set_position((pos[0] - 0.05, pos[1]))
        elif n.startswith("B"):
            graph.node_label_artists[n].set_position((pos[0] + 0.05, pos[1]))
    for e in G.edges:
        pos = graph.edge_label_artists[e].get_position()
    fig.set_facecolor("#eaeaf2")
    return fig
