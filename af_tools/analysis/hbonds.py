import subprocess
import tempfile
import networkx as nx
import re


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

            v1 = f"{matches[0].group(1)}-{matches[0].group(3)}"
            v2 = f"{matches[1].group(1)}-{matches[1].group(3)}"

            graphs[i].add_node(v1, res_name=matches[0].group(2).title())
            graphs[i].add_node(v2, res_name=matches[1].group(2).title())

            if (v1, v2) not in graphs[i].edges:
                graphs[i].add_edge(v1, v2, count=1, saltbridge_count=0)
            else:
                graphs[i].edges[(v1, v2)]["count"] += 1

        for saltbrige_str in saltbridges:
            matches = list(re.finditer(bond_group_ex, saltbrige_str))

            # chain id must be different
            assert matches[0].group(1) != matches[1].group(1)

            v1 = f"{matches[0].group(1)}-{matches[0].group(3)}"
            v2 = f"{matches[1].group(1)}-{matches[1].group(3)}"

            graphs[i].edges[(v1, v2)]["saltbridge_count"] += 1
    return graphs
