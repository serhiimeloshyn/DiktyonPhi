import enum
import subprocess
from typing import Dict, Hashable, Any, Optional, Iterator, Tuple, Iterable, List
import random


class GraphType(enum.Enum):
    """Graph orientation type: directed or undirected."""
    DIRECTED = 0
    UNDIRECTED = 1


class Edge:
    """Representation of an edge between two nodes with associated attributes."""

    def __init__(self, src: 'Node', dest: 'Node', attrs: Dict[str, Any]):
        """
        Initialize an edge from src to dest with given attributes.

        :param src: Source node instance.
        :param dest: Destination node instance.
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

        :param dest: destination node id or Node
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

    # ---------------------- Core mutation API ----------------------

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
        :raises ValueError: If the directed edge already exists.
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

    # ---------------------- Python protocol ----------------------

    def __contains__(self, node_id: Hashable) -> bool:
        """Check whether a node exists in the graph."""
        return node_id in self._nodes

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over Node instances in the graph."""
        return iter(self._nodes.values())

    def __repr__(self):
        edges = self.edge_count()
        return f"Graph({self.type}, nodes: {len(self._nodes)}, edges: {edges})"

    # Task 2: equality of graphs (type, nodes+attrs, edges ignoring edge attrs)
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return False
        if self.type != other.type:
            return False
        # same node ids
        if set(self.node_ids()) != set(other.node_ids()):
            return False
        # node attributes equal
        for nid in self.node_ids():
            if self.node(nid)._attrs != other.node(nid)._attrs:
                return False
        # same set of edges (ignoring edge attributes)
        return self._edge_key_set() == other._edge_key_set()

    # ---------------------- Accessors ----------------------

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

    # ---------------------- Internals ----------------------

    def _create_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """Internal method to create a node."""
        node = Node(self, node_id, attrs if attrs is not None else {})
        self._nodes[node_id] = node
        return node

    def _set_edge(self, src_id: Hashable, target_id: Hashable, attrs: Dict[str, Any]) -> None:
        """Internal method to create a directed edge."""
        if target_id in self._nodes[src_id]._neighbors:
            raise ValueError(f"Edge {src_id}→{target_id} already exists")
        self._nodes[src_id]._neighbors[target_id] = dict(attrs)  # copy to avoid aliasing

    def _edge_key_set(self) -> set:
        """
        Helper for equality: returns a set of edge keys ignoring attributes.
        For DIRECTED: {(u, v), ...}
        For UNDIRECTED: {frozenset({u, v}), ...} (each undirected edge once)
        """
        keys = set()
        if self.type == GraphType.DIRECTED:
            for u in self.node_ids():
                for v in self.node(u).neighbor_ids:
                    keys.add((u, v))
        else:
            seen = set()
            for u in self.node_ids():
                for v in self.node(u).neighbor_ids:
                    if (v, u) in seen:
                        continue
                    seen.add((u, v))
                    keys.add(frozenset((u, v)))
        return keys

    # ---------------------- Export / Visualization ----------------------

    def to_dot(self, label_attr: str = "label", weight_attr: str = "weight") -> str:
        """
        Generate a simple Graphviz (DOT) representation of the graph.
        """
        lines = []
        name = "G"
        connector = "->" if self.type == GraphType.DIRECTED else "--"

        lines.append('digraph {name} {{'.format(name=name) if self.type == GraphType.DIRECTED
                      else 'graph {name} {{'.format(name=name))

        # Nodes
        for node_id in self.node_ids():
            node = self.node(node_id)
            label = node._attrs.get(label_attr, str(node_id))
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
                label = edge._attrs.get(weight_attr, "")
                lines.append(f'    "{node_id}" {connector} "{dst_id}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def export_to_png(self, filename: str) -> None:
        """
        Export the graph to a PNG file using Graphviz (dot). Graphviz must be installed.
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
        """Return SVG representation for IPython notebooks."""
        return self.to_image().data

    def to_image(self):
        """Return graph as SVG (usable in IPython notebook)."""
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
            # e.stderr is already a str because text=True
            raise RuntimeError(f"Graphviz 'dot' command failed: {e} with stderr: {e.stderr}") from e

    # ---------------------- Utilities for tasks ----------------------

    def iter_edges(self, unique_undirected: bool = True) -> Iterator[Tuple[Hashable, Hashable, Dict[str, Any]]]:
        """
        Iterate over edges as (u, v, attrs).

        :param unique_undirected: if True and the graph is UNDIRECTED, each undirected
                                  edge is yielded once with u <= v (by str ordering).
        """
        if self.type == GraphType.DIRECTED or not unique_undirected:
            for u in self.node_ids():
                node = self.node(u)
                for v in node.neighbor_ids:
                    yield (u, v, node._neighbors[v])
        else:
            seen = set()
            for u in self.node_ids():
                node = self.node(u)
                for v in node.neighbor_ids:
                    key = tuple(sorted((u, v), key=lambda x: (str(type(x)), str(x))))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield (u, v, node._neighbors[v])

    def edge_count(self) -> int:
        """Return number of edges m (undirected counted once)."""
        if self.type == GraphType.DIRECTED:
            return sum(node.out_degree for node in self._nodes.values())
        else:
            # count unique undirected
            return sum(1 for _ in self.iter_edges(unique_undirected=True))

    # ---------------------- Task implementations ----------------------

    # Task 1: density for undirected graphs (and generalized for directed too)
    def density(self) -> float:
        """
        Graph density.
        UNDIRECTED: m / (n*(n-1)/2)
        DIRECTED:   m / (n*(n-1))
        For n < 2 returns 0.0.
        """
        n = len(self._nodes)
        if n < 2:
            return 0.0
        m = self.edge_count()
        if self.type == GraphType.UNDIRECTED:
            max_m = n * (n - 1) / 2
        else:
            max_m = n * (n - 1)
        return m / max_m if max_m else 0.0


# ---------------------- Standalone helpers for tasks ----------------------

def random_graph(node_count: int, edge_count: int) -> Graph:
    """
    Task 3: Create a random DIRECTED graph with given numbers of nodes and edges.
    Nodes are labeled 1..node_count. No self-loops. No multi-edges.
    """
    if node_count < 0 or edge_count < 0:
        raise ValueError("node_count and edge_count must be non-negative")
    g = Graph(GraphType.DIRECTED)
    for i in range(1, node_count + 1):
        g.add_node(i, {})

    all_pairs = [(u, v) for u in range(1, node_count + 1) for v in range(1, node_count + 1) if u != v]
    max_possible = len(all_pairs)
    if edge_count > max_possible:
        raise ValueError(f"Too many edges requested: {edge_count} > {max_possible} possible without self-loops")

    chosen = random.sample(all_pairs, edge_count)
    for u, v in chosen:
        g.add_edge(u, v)
    return g


def balanced_nodes_by_weight(g: Graph, weight_attr: str = "weight") -> List[Hashable]:
    """
    Task 4: For a DIRECTED weighted graph, return nodes for which the sum of outgoing
    weights equals the sum of incoming weights. Missing weights are treated as 0.
    """
    if g.type != GraphType.DIRECTED:
        # The task explicitly mentions a directed graph; allow but compute generically.
        pass

    out_sum: Dict[Hashable, float] = {nid: 0.0 for nid in g.node_ids()}
    in_sum: Dict[Hashable, float] = {nid: 0.0 for nid in g.node_ids()}

    for u, v, attrs in g.iter_edges(unique_undirected=False):
        w = float(attrs.get(weight_attr, 0.0))
        out_sum[u] = out_sum.get(u, 0.0) + w
        in_sum[v] = in_sum.get(v, 0.0) + w

    balanced = [nid for nid in g.node_ids() if abs(out_sum.get(nid, 0.0) - in_sum.get(nid, 0.0)) == 0.0]
    return balanced


def remove_negative_weights(og: Graph, weight_attr: str = "weight") -> Graph:
    """
    Task 5: For an UNDIRECTED weighted graph, return a new graph with all edges with
    negative weight removed. Node attributes are preserved; edge attributes are preserved
    for non-negative edges.
    """
    if og.type != GraphType.UNDIRECTED:
        raise ValueError("remove_negative_weights expects an UNDIRECTED weighted graph")

    ng = Graph(GraphType.UNDIRECTED)
    # copy nodes with attributes
    for nid in og.node_ids():
        ng.add_node(nid, dict(og.node(nid)._attrs))

    # copy edges with non-negative weight
    for u, v, attrs in og.iter_edges(unique_undirected=True):
        w = float(attrs.get(weight_attr, 0.0))
        if w >= 0.0:
            ng.add_edge(u, v, dict(attrs))
    return ng


# ---------------------- Demo / simple tests ----------------------
if __name__ == "__main__":
    # Task 1 demo
    g = Graph(GraphType.UNDIRECTED)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 1)
    print("Density (should be 4/6 ≈ 0.6667):", g.density())

    # Task 2 demo
    g2 = Graph(GraphType.UNDIRECTED)
    g2.add_edge(2, 3)
    g2.add_edge(3, 4)
    g2.add_edge(4, 1)
    g2.add_edge(1, 2)
    print("Equality g == g2 (should be True):", g == g2)

    # Task 3 demo
    rg = random_graph(5, 7)
    print("Random graph:", rg)

    # Task 4 demo
    dg = Graph(GraphType.DIRECTED)
    dg.add_edge(1, 2, dict(weight=1))
    dg.add_edge(2, 3, dict(weight=2))
    dg.add_edge(3, 4, dict(weight=1))
    dg.add_edge(4, 1, dict(weight=1))
    print("Balanced nodes by weight (should list 1 and 4):", balanced_nodes_by_weight(dg))

    # Task 5 demo
    og = Graph(GraphType.UNDIRECTED)
    og.add_edge(1, 2, dict(weight=1))
    og.add_edge(2, 3, dict(weight=-1))
    og.add_edge(3, 4, dict(weight=-2))
    og.add_edge(4, 1, dict(weight=0))
    ng = remove_negative_weights(og)
    print("Original edges:", sorted((min(u, v), max(u, v)) for u, v, _ in og.iter_edges(True)))
    print("Filtered  edges:", sorted((min(u, v), max(u, v)) for u, v, _ in ng.iter_edges(True)))
