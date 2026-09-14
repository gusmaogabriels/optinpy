"""Native shortest-path algorithms on optinpy graphs (node 0 is artificial)."""
from heapq import heappop, heappush
from itertools import count
from math import isfinite


def _edges(network):
    return [(u, v, a.cost) for u, arcs in network.arcs.items() for v, a in arcs.items()
            if u != 0 and v != 0]


def _initial(network, n0):
    if n0 == 0 or n0 not in network.nodes:
        raise ValueError("The source must be an existing non-artificial node")
    distances = dict.fromkeys(network.nodes, float('inf'))
    predecessors = dict.fromkeys(network.nodes)
    distances[n0], predecessors[n0] = 0., n0
    return predecessors, distances


def fmb(network, n0, verbose=False):
    """Bellman-Ford; reject a negative cycle reachable from the source."""
    predecessors, distances = _initial(network, n0)
    edges = _edges(network)
    for _ in range(max(0, len(network.nodes) - 2)):
        changed = False
        for u, v, cost in edges:
            if distances[u] + cost < distances[v]:
                distances[v], predecessors[v] = distances[u] + cost, u
                changed = True
        if verbose:
            print(distances)
        if not changed:
            break
    if any(distances[u] + cost < distances[v] for u, v, cost in edges):
        raise ValueError("A reachable negative-weight cycle exists")
    return predecessors, distances


def dijkstra(network, n0, verbose=False):
    """Heap-based Dijkstra; supports mixed labels without comparing them."""
    predecessors, distances = _initial(network, n0)
    edges = _edges(network)
    if any(cost < 0 or not isfinite(cost) for _, _, cost in edges):
        raise ValueError("Dijkstra requires finite nonnegative edge costs")
    serial = count()
    queue = [(0., next(serial), n0)]
    while queue:
        distance, _, u = heappop(queue)
        if distance != distances[u]:
            continue
        for v, arc in network.arcs.get(u, {}).items():
            if v == 0:
                continue
            candidate = distance + arc.cost
            if candidate < distances[v]:
                distances[v], predecessors[v] = candidate, u
                heappush(queue, (candidate, next(serial), v))
        if verbose:
            print(distances)
    return predecessors, distances
