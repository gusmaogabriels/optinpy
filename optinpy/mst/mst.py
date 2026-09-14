"""Native minimum spanning trees; directed input arcs are treated as undirected."""
from math import isfinite


def _data(network):
    nodes = [u for u in network.nodes if u != 0]
    edges = [(a.cost, u, v) for u, arcs in network.arcs.items() for v, a in arcs.items()
             if u != 0 and v != 0 and u != v and isfinite(a.cost)]
    return nodes, edges


class _Components:
    def __init__(self, nodes):
        self.parent = {u: u for u in nodes}

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def join(self, u, v):
        u, v = self.find(u), self.find(v)
        if u == v:
            return False
        self.parent[v] = u
        return True


def prim(network, n0, verbose=False):
    """Prim's cut-growing algorithm; raises on disconnected graphs."""
    nodes, edges = _data(network)
    if n0 not in nodes:
        raise ValueError("The start must be an existing non-artificial node")
    visited, result = {n0}, []
    while len(visited) < len(nodes):
        candidates = [edge for edge in edges if (edge[1] in visited) != (edge[2] in visited)]
        if not candidates:
            raise ValueError("The graph is disconnected")
        _, u, v = min(candidates, key=lambda edge: edge[0])
        result.append([u, v])
        visited.update((u, v))
        if verbose:
            print(result)
    return result


def kruskal(network, verbose=False):
    """Kruskal's sorted-edge algorithm with union-find."""
    nodes, edges = _data(network)
    components, result = _Components(nodes), []
    for _, u, v in sorted(edges, key=lambda edge: edge[0]):
        if components.join(u, v):
            result.append([u, v])
            if verbose:
                print(result)
    if len(result) != max(0, len(nodes) - 1):
        raise ValueError("The graph is disconnected")
    return result


def boruvka(network, verbose=False):
    """Boruvka's component-wise cheapest-edge merging algorithm."""
    nodes, edges = _data(network)
    components, result = _Components(nodes), []
    while len(result) < len(nodes) - 1:
        cheapest = {}
        for edge in edges:
            cost, u, v = edge
            ru, rv = components.find(u), components.find(v)
            if ru == rv:
                continue
            for root in (ru, rv):
                if root not in cheapest or cost < cheapest[root][0]:
                    cheapest[root] = edge
        if not cheapest:
            raise ValueError("The graph is disconnected")
        for _, u, v in cheapest.values():
            if components.join(u, v):
                result.append([u, v])
        if verbose:
            print(result)
    return result
