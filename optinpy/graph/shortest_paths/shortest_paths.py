# -*- coding: utf-8 -*-

from ..base import argparser, graph 

def bfm(network,n0):
    
    return 

def dijkstra(network,n0):
    nodes = network.nodes.keys()
    arcs = network.arcs
    ds = [float('inf') for i in range(0,len(nodes))] # distances list
    rot = [0 for i in range(0,len(nodes))] # route list
    on = list(nodes) # open nodes list (O)
    cn = [on.pop(on.index(n0))] # closed nodes list (F)
    ds[nodes.index(n0)] = 0
    r = n0
    while len(on)>0:
        node = network.nodes[r]
        for i in node.children:
            if ds[nodes.index(i)] > network.costs[arcs.index([r,i])] + ds[nodes.index(r)]:
                ds[nodes.index(i)] = network.costs[arcs.index([r,i])] + ds[nodes.index(r)]
                rot[nodes.index(i)] = r
            else:
                pass
        r = min(zip([ds[nodes.index(i)] for i in on], on))[1]
        cn += [on.pop(on.index(r))]
    return rot, ds