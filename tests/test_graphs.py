import math

import numpy as np
import pytest

import optinpy as op


def graph(edges):
    network = op.graph()
    for edge in edges:
        network.add_connection(*edge)
    return network


@pytest.mark.parametrize('method', ['dijkstra', 'fmb'])
def test_shortest_distances_and_predecessors(method):
    n = graph([('A', 1, 2), ('A', 'C', 4), (1, 'C', 1)])
    n.add_node('unreachable', 0)
    pred, dist = getattr(op.sp, method)(n, 'A')
    assert dist['C'] == 3 and pred['C'] == 1 and pred[1] == 'A'
    assert math.isinf(dist['unreachable']) and pred['unreachable'] is None
    assert math.isinf(dist[0])


def test_negative_edges_and_cycles():
    n = graph([('A', 'B', 2), ('B', 'C', -3), ('A', 'C', 1)])
    assert op.sp.fmb(n, 'A')[1]['C'] == -1
    with pytest.raises(ValueError, match='nonnegative'):
        op.sp.dijkstra(n, 'A')
    n.add_connection('C', 'B', 1)
    with pytest.raises(ValueError, match='negative-weight cycle'):
        op.sp.fmb(n, 'A')


@pytest.mark.parametrize('method', ['prim', 'kruskal', 'boruvka'])
def test_mst_cost_and_exclusion_of_artificial_arcs(method):
    n = graph([('A', 'B', 7), ('A', 'D', 5), ('B', 'C', 8), ('B', 'D', 9),
               ('B', 'E', 7), ('C', 'E', 5), ('D', 'E', 15), ('D', 'F', 6),
               ('E', 'F', 8), ('E', 'G', 9), ('F', 'G', 11)])
    edges = op.mst.prim(n, 'A') if method == 'prim' else getattr(op.mst, method)(n)
    assert len(edges) == 6
    assert all(0 not in edge for edge in edges)
    assert sum(n.arcs[u][v].cost for u, v in edges) == 39
    n.add_node('isolated', 0)
    with pytest.raises(ValueError, match='disconnected'):
        op.mst.prim(n, 'A') if method == 'prim' else getattr(op.mst, method)(n)


def test_bidirectional_connections():
    n = op.graph()
    n.add_connection('A', 'B', 2, bidirectional=True)
    assert op.sp.dijkstra(n, 'B')[1]['A'] == 2


def test_minimum_cost_flow_balance_and_cost():
    n = op.graph()
    n.add_node(['s', 'm', 't'], [3, 0, -3])
    for edge in [('s', 'm', 1), ('m', 't', 2), ('s', 't', 5)]:
        n.add_connection(*edge)
    result = op.mcfp.bigm(n, 100)
    assert result is n
    assert n.solution['success']
    assert n.solution['f'] == 9
    for key in ['s', 'm', 't']:
        outgoing = sum(a.flow for a in n.arcs.get(key, {}).values())
        incoming = sum(arcs[key].flow for arcs in n.arcs.values() if key in arcs)
        np.testing.assert_allclose(outgoing-incoming, n.nodes[key].b, atol=1e-6)


def test_infeasible_flow_fails():
    n = op.graph()
    n.add_node(['s', 't'], [1, -1])
    with pytest.raises(ValueError, match='Artificial flow'):
        op.mcfp.bigm(n, 100)


def test_original_readme_minimum_cost_flow_example():
    n = op.graph()
    for key, balance in zip(range(1, 6), [10, 4, 0, -6, -8]):
        n.add_node(key, balance)
    for edge in [(1, 2, 1), (1, 3, 8), (1, 4, 1), (2, 3, 2),
                 (3, 4, 1), (3, 5, 4), (4, 5, 12), (5, 2, 7)]:
        n.add_connection(*edge)
    op.mcfp.bigm(n, 20)
    assert n.solution['success']
    np.testing.assert_allclose(n.solution['f'], 58., atol=1e-7)
    for key in range(1, 6):
        outgoing = sum(a.flow for a in n.arcs.get(key, {}).values())
        incoming = sum(arcs[key].flow for arcs in n.arcs.values() if key in arcs)
        np.testing.assert_allclose(outgoing-incoming, n.nodes[key].b, atol=1e-7)
