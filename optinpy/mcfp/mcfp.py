"""Minimum-cost flow using optinpy's own Big-M tableau formulation."""
import math

import jax.numpy as jnp

from ..simplex.base import simplex


def bigm(network, mcost, verbose=False, *, max_iter=10000, tol=1e-6):
    """Solve nonnegative, uncapacitated flow with artificial root arc cost M.

    Real node supplies must sum to zero. Raises if the LP fails or retains
    artificial flow (infeasible network or M too small). Flow values are written
    to the graph's arcs and the LP diagnostics to ``network.solution``.
    """
    if not math.isfinite(mcost) or mcost <= 0 or tol <= 0:
        raise ValueError("mcost and tol must be finite and positive")
    nodes = [key for key in network.nodes if key != 0]
    if not nodes:
        raise ValueError("The network must contain real nodes")
    supply = jnp.asarray([network.nodes[key].b for key in nodes], dtype=float)
    if abs(float(jnp.sum(supply))) > tol:
        raise ValueError("Node supplies and demands must sum to zero")
    arcs = [arc for outgoing in network.arcs.values() for arc in outgoing.values()]
    costs = jnp.asarray([mcost if 0 in (a.source.key, a.destination.key) else a.cost for a in arcs])
    incidence = jnp.asarray([[float(a.source.key == key) - float(a.destination.key == key)
                              for a in arcs] for key in nodes])
    lp = simplex(jnp.concatenate((incidence, -incidence)),
                 jnp.concatenate((supply, -supply)), costs, tol=tol)
    result = lp.solve(max_iter=max_iter)
    if not result['success']:
        raise ValueError(f"Minimum-cost flow solve failed (simplex status {result['status']})")
    flows = result['x'].tolist()
    artificial = sum(flow for a, flow in zip(arcs, flows)
                     if 0 in (a.source.key, a.destination.key))
    if artificial > tol:
        raise ValueError("Artificial flow remains: the network is infeasible or mcost is too small")
    network.root_weight = mcost
    for arc, flow in zip(arcs, flows):
        arc.flow = float(flow)
        if 0 in (arc.source.key, arc.destination.key):
            arc.cost = mcost
    network.solution = result
    if verbose:
        print(f"Minimum flow cost: {float(result['f'])}; pivots: {result['iterations']}")
    return network
