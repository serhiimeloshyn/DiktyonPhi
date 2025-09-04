import enum
import subprocess
from typing import Dict, Hashable, Any, Optional, Iterator, Tuple


class GraphType(enum.Enum):
    """Graph orientation type: directed or undirected."""
    DIRECTED = 0
    UNDIRECTED = 1


class Edge:
    """Representation of an edge between two nodes with associated attributes."""

    def __init__(self, src: 'Node', dest: 'Node', attrs: Dict[str, Any]):
        """
        Initialize an edge from src_id to dest_id with given attributes.

        :param src: Source node identifier.
        :param dest: Destination node identifier.
        :param attrs: Dictionary of edge attributes.
        """
        self.src = src
        self.dest = dest
        self._attrs = attrs

    def __getitem__(self, key: str) -> Any:
        """Access edge attribute by key."""
        return self._attrs[key]

    def __setitem__(self, key: str, val: Any) -> None:
        """Set edge attribute by key."""
        self._attrs[key] = val

    def __repr__(self):
        return f"Edge({self.src.id}→{self.dest.id}, {self._attrs})"


class Node:
    """Representation of a graph node with attributes and outgoing edges."""

    def __init__(self, graph: 'Graph', node_id: Hashable, attrs: Dict[str, Any]):
        """
        Initialize a node with a given identifier and attributes.

        :param node_id: Unique identifier of the node.
        :param attrs: Dictionary of node attributes.
        """
        self.id = node_id
        self.graph = graph
        self._attrs = attrs
        self._neighbors: Dict[Hashable, Dict[str, Any]] = {}

    def __getitem__(self, item: str) -> Any:
        """Access node attribute by key."""
        return self._attrs[item]

    def __setitem__(self, item: str, val: Any) -> None:
        """Set node attribute by key."""
        self._attrs[item] = val

    def to(self, dest: Hashable | 'Node') -> Edge:
        """
        Get the edge from this node to the specified destination node.

        :param dest_id: ID of the target node.
        :return: Edge instance representing the connection.
        :raises ValueError: If no such edge exists.
        """
        dest_id = dest.id if isinstance(dest, Node) else dest
        if dest_id not in self._neighbors:
            raise ValueError(f"No edge from {self.id} to {dest_id}")
        return Edge(self, self.graph.node(dest_id), self._neighbors[dest_id])

    def connect_to(self,  dest: Hashable | 'Node', attrs: Optional[Dict[str, Any]] = None):
        dest = dest if isinstance(dest, Node) else self.graph.node(dest)
        assert dest.graph == self.graph, f"Destination node {dest.id} is not in the same graph"
        assert dest.id in self.graph, f"Destination node {dest.id} is not in graph"
        self.graph.add_edge(self.id, dest.id, attrs if attrs is not None else {})

    def is_edge_to(self, dest: Hashable | 'Node') -> bool:
        """
        Check if this node has an edge to the given node.

        :param dest_id: ID of the target node.
        :return: True if edge exists, False otherwise.
        """
        dest_id = dest.id if isinstance(dest, Node) else dest
        return dest_id in self._neighbors

    @property
    def neighbor_ids(self) -> Iterator[Hashable]:
        """Return an iterator over IDs of neighboring nodes."""
        return iter(self._neighbors)

    @property
    def neighbor_nodes(self) -> Iterator['Node']:
        for id in self.neighbor_ids:
            yield self.graph.node(id)

    @property
    def out_degree(self) -> int:
        """Return the number of outgoing edges."""
        return len(self._neighbors)

    def __repr__(self):
        return f"Node({self.id}, {self._attrs})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Graph:
    """Graph data structure supporting directed and undirected graphs."""

    def __init__(self, type: GraphType):
        """
        Initialize a graph with the given type.

        :param type: GraphType.DIRECTED or GraphType.UNDIRECTED
        """
        self.type = type
        self._nodes: Dict[Hashable, Node] = {}

    def add_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """
        Add a new node to the graph.

        :param node_id: Unique node identifier.
        :param attrs: Optional dictionary of attributes.
        :raises ValueError: If the node already exists.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists")
        return self._create_node(node_id, attrs if attrs is not None else {})

    def add_edge(self, src_id: Hashable, dst_id: Hashable,
                 attrs: Optional[Dict[str, Any]] = None) -> Tuple[Node, Node]:
        """
        Add a new edge to the graph. Nodes are created automatically if missing.

        :param src_id: Source node ID.
        :param dst_id: Destination node ID.
        :param attrs: Optional dictionary of edge attributes.
        :raises ValueError: If the edge already exists.
        """
        attrs = attrs if attrs is not None else {}
        if src_id not in self._nodes:
            self._create_node(src_id, {})
        if dst_id not in self._nodes:
            self._create_node(dst_id, {})
        self._set_edge(src_id, dst_id, attrs)
        if self.type == GraphType.UNDIRECTED:
            self._set_edge(dst_id, src_id, attrs)
        return (self._nodes[src_id], self._nodes[dst_id])

    def __contains__(self, node_id: Hashable) -> bool:
        """Check whether a node exists in the graph."""
        return node_id in self._nodes

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over node IDs in the graph."""
        return iter(self._nodes.values())

    def node_ids(self) -> Iterator[Hashable]:
        return iter(self._nodes.keys())

    def node(self, node_id: Hashable) -> Node:
        """
        Get the Node instance with the given ID.

        :param node_id: The ID of the node.
        :return: Node instance.
        :raises KeyError: If the node does not exist.
        """
        return self._nodes[node_id]

    def _create_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """Internal method to create a node."""
        node = Node(self, node_id, attrs)
        self._nodes[node_id] = node
        return node

    def _set_edge(self, src_id: Hashable, target_id: Hashable, attrs: Dict[str, Any]) -> None:
        """Internal method to create a directed edge."""
        if target_id in self._nodes[src_id]._neighbors:
            raise ValueError(f"Edge {src_id}→{target_id} already exists")
        self._nodes[src_id]._neighbors[target_id] = attrs

    def __repr__(self):
        edges = sum(node.out_degree for node in self._nodes.values())
        if self.type == GraphType.UNDIRECTED:
            edges //= 2
        return f"Graph({self.type}, nodes: {len(self._nodes)}, edges: {edges})"

    def to_dot(self, label_attr:str ="label", weight_attr:str = "weight") -> str:
        """
        Generate a simple Graphviz (DOT) representation of the graph. Generated by ChatGPT.

        :return: String in DOT language.
        """
        lines = []
        name = "G"
        connector = "->" if self.type == GraphType.DIRECTED else "--"

        lines.append(f'digraph {name} {{' if self.type == GraphType.DIRECTED else f'graph {name} {{')

        # Nodes
        for node_id in self.node_ids():
            node = self.node(node_id)
            label = node[label_attr] if label_attr in node._attrs else str(node_id)
            lines.append(f'    "{node_id}" [label="{label}"];')

        # Edges
        seen = set()
        for node_id in self.node_ids():
            node = self.node(node_id)
            for dst_id in node.neighbor_ids:
                if self.type == GraphType.UNDIRECTED and (dst_id, node_id) in seen:
                    continue
                seen.add((node_id, dst_id))
                edge = node.to(dst_id)
                label = edge[weight_attr] if weight_attr in edge._attrs else ""
                lines.append(f'    "{node_id}" {connector} "{dst_id}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)


    def export_to_png(self, filename: str = None) -> None:
        """
        Export the graph to a PNG file using Graphviz (dot). Graphviz (https://graphviz.org/)
         must be installed.

        :param filename: Output PNG filename.
        :raises RuntimeError: If Graphviz 'dot' command fails.
        """
        dot_data = self.to_dot()
        try:
            subprocess.run(
                ["dot", "-Tpng", "-o", filename],
                input=dot_data,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz 'dot' command failed: {e}") from e

    def _repr_svg_(self):
        """
          Return SVG representation of the graph for Jupyter notebook (implementation
          of protocol of IPython).
        """
        return self.to_image().data

    def to_image(self):
        """
            Return graph as SVG (usable in IPython notebook).
        """
        from IPython.display import SVG
        dot_data = self.to_dot()
        try:
            process = subprocess.run(
                ['dot', '-Tsvg'],
                input=dot_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return SVG(data=process.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz 'dot' command failed: {e} with stderr: {e.stderr.decode('utf-8')}") from e

if __name__ == "__main__":
    # Create a directed graph
    g = Graph(GraphType.DIRECTED)

    # Add nodes with attributes
    g.add_node("A", {"label": "Start", "color": "green"})
    g.add_node("B", {"label": "Middle", "color": "yellow"})
    g.add_node("C", {"label": "End", "color": "red"})
    g.add_node("D", {"label": "Optional", "color": "blue"})

    # Add edges with attributes
    g.add_edge("A", "B", {"weight": 1.0, "type": "normal"})
    g.add_edge("B", "C", {"weight": 2.5, "type": "critical"})
    g.add_edge("A", "D", {"weight": 0.8, "type": "optional"})
    g.add_edge("D", "C", {"weight": 1.7, "type": "fallback"})

    # Access and update node attribute
    print("Node A color:", g.node("A")["color"])
    g.node("A")["color"] = "darkgreen"

    # Access edge and modify its weight
    edge = g.node("A").to("B")
    print("Edge A→B weight:", edge["weight"])
    edge["weight"] = 1.1

    # Iterate through the graph
    print("\nGraph structure:")
    for node_id in g.node_ids():
        node = g.node(node_id)
        print(f"Node {node.id}: label={node['label']}, out_degree={node.out_degree}")
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            print(f"  → {neighbor_id} (weight={edge['weight']}, type={edge['type']})")

    print("-----------------")
    print(g.to_image())




from diktyonphi import Graph, GraphType

def pocet_uzlu(g):
    return len(g)

def pocet_hran(g):
    hrany = sum(node.out_degree for node in g)
    return hrany if g.type == GraphType.DIRECTED else hrany // 2

def vypis_hrany_s_atributy(g):
    for node in g:
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            print(f"{node.id} → {neighbor_id}: váha = {edge['weight']}, typ = {edge['type']}")

#pouzivame graf ktery uz mame
if __name__ == "__main__":
    g = Graph(GraphType.DIRECTED)

    #graf z diktyonphi.py
    g.add_node("A", {"label": "Start", "color": "green"})
    g.add_node("B", {"label": "Middle", "color": "yellow"})
    g.add_node("C", {"label": "End", "color": "red"})
    g.add_node("D", {"label": "Optional", "color": "blue"})

    g.add_edge("A", "B", {"weight": 1.0, "type": "normal"})
    g.add_edge("B", "C", {"weight": 2.5, "type": "critical"})
    g.add_edge("A", "D", {"weight": 0.8, "type": "optional"})
    g.add_edge("D", "C", {"weight": 1.7, "type": "fallback"})

    # Analyza:
    print("Počet uzlů:", pocet_uzlu(g))
    print("Počet hran:", pocet_hran(g))
    print("Hrany s atributy:")
    vypis_hrany_s_atributy(g)

#Vyberte všechny hrany typu 'critical'
def hrany_dle_typu(g, hledany_typ):
    return [(node.id, neighbor_id)
            for node in g
            for neighbor_id in node.neighbor_ids
            if node.to(neighbor_id)['type'] == hledany_typ]

def vypis_hrany_dle_typu(g, hledany_typ):
    hrany = hrany_dle_typu(g, hledany_typ)
    print(f"Hrany typu '{hledany_typ}':")
    for src, dst in hrany:
        print(f"{src} → {dst}")

#test:
vypis_hrany_dle_typu(g, "critical")

#2: Celková váha všech hran
def celkova_vaha_hran(g):
    return sum(
        node.to(neighbor_id)["weight"]
        for node in g
        for neighbor_id in node.neighbor_ids
        if "weight" in node.to(neighbor_id)._attrs
    )

def vypis_celkovou_vahu(g):
    suma = celkova_vaha_hran(g)
    print(f"Celková váha všech hran: {suma}")

#test:
vypis_celkovou_vahu(g)

#3:Seřadit hrany podle váhy sestupně
def serad_hrany_podle_vahy(g):
    hrany = []
    for node in g:
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            hrany.append((node.id, neighbor_id, edge["weight"]))
    return sorted(hrany, key=lambda x: -x[2])

def vypis_serazene_hrany(g):
    hrany = serad_hrany_podle_vahy(g)
    print("Hrany seřazené podle váhy:")
    for src, dst, w in hrany:
        print(f"{src} → {dst}: {w}")

#test: použít na graf z diktyonphi.py
vypis_serazene_hrany(g)


#4:Najít hrany bez atributu ‚weight‘
def hrany_bez_vahy(g):
    return [
        (node.id, neighbor_id)
        for node in g
        for neighbor_id in node.neighbor_ids
        if "weight" not in node.to(neighbor_id)._attrs
    ]

def vypis_hrany_bez_vahy(g):
    hrany = hrany_bez_vahy(g)
    if hrany:
        print("Hrany bez atributu 'weight':")
        for src, dst in hrany:
            print(f"{src} → {dst}")
    else:
        print("Všechny hrany mají atribut 'weight'")

#test: použít na graf z diktyonphi.py
vypis_hrany_bez_vahy(g)

#2: Stupně uzlů (výstupní stupeň, izolace, průměr)
#1: Výpis výstupních stupňů všech uzlů
def vypis_stupne_uzlu(g):
    print("Výstupní stupně uzlů:")
    for node in g:
        print(f"{node.id} → stupeň: {node.out_degree}")

#test:
vypis_stupne_uzlu(g)


#2:Seznam uzlů se stupněm 0
def uzly_s_nulovym_stupnem(g):
    return [node.id for node in g if node.out_degree == 0]

def vypis_uzly_s_nulovym_stupnem(g):
    nulove = uzly_s_nulovym_stupnem(g)
    if nulove:
        print("Uzly se stupněm 0:", ", ".join(nulove))
    else:
        print("Žádné uzly se stupněm 0")

#test:
vypis_uzly_s_nulovym_stupnem(g)


#3:Průměrný výstupní stupeň grafu
def prumerny_stupen(g):
    return sum(node.out_degree for node in g) / len(g)

def vypis_prumerny_stupen(g):
    print(f"Průměrný výstupní stupeň: {prumerny_stupen(g):.2f}")

#test:
vypis_prumerny_stupen(g)


#4:Uzel s nejvyšším výstupním stupněm
def max_stupen(g):
    return max(((node.out_degree, node.id) for node in g), default=(0, None))

def vypis_max_stupen(g):
    stupen, uzel = max_stupen(g)
    print(f"Nejvyšší výstupní stupeň má uzel {uzel} → {stupen}")

#test:
vypis_max_stupen(g)

#3: Smyčky v grafu (uzly, které vedou sami na sebe)
#1:Existuje v grafu alespoň jedna smyčka?
def existuje_smycka(g):
    return any(node.is_edge_to(node.id) for node in g)

def vypis_existenci_smycky(g):
    if existuje_smycka(g):
        print("Graf obsahuje smyčku.")
    else:
        print("Graf neobsahuje žádné smyčky.")

#test:
vypis_existenci_smycky(g)


#2:Vypsat všechny uzly, které mají smyčku
def uzly_se_smyckou(g):
    return [node.id for node in g if node.is_edge_to(node.id)]

def vypis_uzly_se_smyckou(g):
    uzly = uzly_se_smyckou(g)
    if uzly:
        print("Uzly se smyčkou:", ", ".join(uzly))
    else:
        print("Žádné uzly nemají smyčku.")

#test:
vypis_uzly_se_smyckou(g)

#4: Záporné hrany v grafu

#1:Existuje alespoň jedna hrana se zápornou váhou?
def existuje_zaporna_hrana(g):
    for node in g:
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            if "weight" in edge._attrs and edge["weight"] < 0:
                return True
    return False

def vypis_existenci_zaporne_hrany(g):
    if existuje_zaporna_hrana(g):
        print("Graf obsahuje alespoň jednu zápornou hranu.")
    else:
        print("Žádné záporné hrany nebyly nalezeny.")

#test:
vypis_existenci_zaporne_hrany(g)


#Vypsat všechny záporné hrany
def zaporne_hrany(g):
    return [
        (node.id, neighbor_id, edge["weight"])
        for node in g
        for neighbor_id in node.neighbor_ids
        if "weight" in node.to(neighbor_id)._attrs and node.to(neighbor_id)["weight"] < 0
    ]

def vypis_zaporne_hrany(g):
    hrany = zaporne_hrany(g)
    if hrany:
        print("Záporné hrany:")
        for src, dst, w in hrany:
            print(f"{src} → {dst}: váha = {w}")
    else:
        print("Žádné záporné hrany nejsou.")

#test:
vypis_zaporne_hrany(g)


#5:Linearita grafu (je graf lineární řetězec?)
#1:Je graf lineární?
#Lineární graf má uzly se stupněm max. 2 (nejčastěji: 2, 2, 2, ..., 1, 1)
def je_linearni(g):
    for node in g:
        deg = node.out_degree
        if g.type == GraphType.UNDIRECTED:
            deg = len(list(node.neighbor_ids))
        if deg > 2:
            return False
    return True

def vypis_linearitu(g):
    if je_linearni(g):
        print("Graf je lineární (řetězec).")
    else:
        print("Graf není lineární.")

#test:
vypis_linearitu(g)

#6: Cesty, cykly, souvislost
#1:Najít všechny cesty z A do B
def najdi_vsechny_cesty(g, start_id, end_id, path=None):
    path = path or []
    path = path + [start_id]
    if start_id == end_id:
        return [path]
    if start_id not in g:
        return []
    paths = []
    for neighbor in g.node(start_id).neighbor_ids:
        if neighbor not in path:
            nove = najdi_vsechny_cesty(g, neighbor, end_id, path)
            paths.extend(nove)
    return paths

def vypis_vsechny_cesty(g, start_id, end_id):
    cesty = najdi_vsechny_cesty(g, start_id, end_id)
    print(f"Cesty z {start_id} do {end_id}:")
    if cesty:
        for cesta in cesty:
            print(" → ".join(cesta))
    else:
        print("Žádná cesta nenalezena.")

#test:
vypis_vsechny_cesty(g, "A", "C")


#2:Obsahuje graf cyklus?
def existuje_cyklus(g):
    visited = set()
    stack = set()

    def dfs(node_id):
        visited.add(node_id)
        stack.add(node_id)
        for neighbor_id in g.node(node_id).neighbor_ids:
            if neighbor_id not in visited:
                if dfs(neighbor_id):
                    return True
            elif neighbor_id in stack:
                return True
        stack.remove(node_id)
        return False

    for node in g:
        if node.id not in visited:
            if dfs(node.id):
                return True
    return False

def vypis_existenci_cyklu(g):
    if existuje_cyklus(g):
        print("Graf obsahuje cyklus.")
    else:
        print("Graf je acyklický.")

#test:
vypis_existenci_cyklu(g)

#3:Je graf souvislý? (funguje pro neorientovaný graf)
def je_souvisly(g):
    if not g._nodes:
        return True

    start_id = next(iter(g.node_ids()))
    navstivene = set()

    def dfs(uzel_id):
        navstivene.add(uzel_id)
        for soused in g.node(uzel_id).neighbor_ids:
            if soused not in navstivene:
                dfs(soused)

    dfs(start_id)
    return len(navstivene) == len(g)

def vypis_souvislost(g):
    if je_souvisly(g):
        print("Graf je souvislý.")
    else:
        print("Graf není souvislý.")

#test:
vypis_souvislost(g)

#7: Práce s atributy uzlů a hran
#1:Najít uzly podle atributu (např. color == "red")
def uzly_podle_atributu(g, attr_key, attr_value):
    return [node.id for node in g if node._attrs.get(attr_key) == attr_value]

def vypis_uzly_podle_barvy(g, barva):
    uzly = uzly_podle_atributu(g, "color", barva)
    if uzly:
        print(f"Uzly s barvou '{barva}':", ", ".join(uzly))
    else:
        print(f"Žádné uzly s barvou '{barva}' nebyly nalezeny.")

#test:
vypis_uzly_podle_barvy(g, "red")


#2:Zvýšit váhu všech hran o hodnotu N
def zvysit_vahy_hran(g, o_kolik=1.0):
    for node in g:
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            if "weight" in edge._attrs:
                edge["weight"] += o_kolik

def vypis_hran_po_upravach(g):
    print("Upravené váhy hran:")
    vypis_hrany_s_atributy(g)

#test:
zvysit_vahy_hran(g, 1.0)
vypis_hran_po_upravach(g)


#3: Najít hrany určitého typu (např. "optional")
def hrany_dle_typu_na_odstraneni(g, odstran_typ):
    return [(node.id, neighbor_id)
            for node in g
            for neighbor_id in node.neighbor_ids
            if node.to(neighbor_id)['type'] == odstran_typ]

def vypis_hrany_na_odstraneni(g, odstran_typ):
    hrany = hrany_dle_typu_na_odstraneni(g, odstran_typ)
    if hrany:
        print(f"Hrany typu '{odstran_typ}', které lze logicky odstranit:")
        for src, dst in hrany:
            print(f"{src} → {dst}")
    else:
        print(f"Žádné hrany typu '{odstran_typ}' nebyly nalezeny.")

#test:
vypis_hrany_na_odstraneni(g, "optional")

#8: Vizualizace a export grafu
#Zobrazit graf jako SVG (v prostředí IPython / Jupyter)
def zobrazit_svg(g):
    try:
        from IPython.display import display
        svg = g.to_image()
        display(svg)
    except Exception as e:
        print("Nelze zobrazit SVG:", e)

#Pokud běžíš v Jupyteru, zavolej: zobrazit_svg(g)


#Export grafu jako PNG soubor (vyžaduje Graphviz)
def exportuj_png(g, filename="graf.png"):
    try:
        g.export_to_png(filename)
        print(f"Graf byl exportován jako PNG do souboru: {filename}")
    except Exception as e:
        print("Chyba při exportu PNG:", e)

#test:
exportuj_png(g, "graf_zkouska.png")


#Vypsat DOT reprezentaci grafu
def vypis_dot(g):
    print("DOT reprezentace grafu:")
    print(g.to_dot())

#test:
vypis_dot(g)


#Uložit DOT popis do souboru
def uloz_dot_do_souboru(g, filename="graf.dot"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(g.to_dot())
        print(f"DOT soubor uložen jako: {filename}")
    except Exception as e:
        print("Chyba při ukládání DOT souboru:", e)

#test:
uloz_dot_do_souboru(g, "graf_zkouska.dot")
