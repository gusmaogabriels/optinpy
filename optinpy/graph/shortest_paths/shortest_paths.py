# -*- coding: utf-8 -*-

from ..base import argparser, graph 

def bfm(network,n0):
    
    return 

def dijkstra(network,n0,verbose=False):
    nodes = network.nodes.keys()
    arcs = network.arcs
    ds = [float('inf') for i in range(0,len(nodes))] # distances list
    rot = [0 for i in range(0,len(nodes))] # route list
    on = list(nodes) # open nodes list (O)
    cn = [on.pop(on.index(n0))] # closed nodes list (F)
    ds[nodes.index(n0)] = 0
    r = n0
    t = 0
    while len(on)>0:
        r0 = r
        node = network.nodes[r]
        for i in node.children:
            if ds[nodes.index(i)] > network.costs[arcs.index([r,i])] + ds[nodes.index(r)]:
                ds[nodes.index(i)] = network.costs[arcs.index([r,i])] + ds[nodes.index(r)]
                rot[nodes.index(i)] = r
            else:
                pass
        r = min(zip([ds[nodes.index(i)] for i in on], on))[1]
        cn += [on.pop(on.index(r))]
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('r0: {}'.format(r0))
            print('d: {}'.format(ds))
            print('rot: {}'.format(rot))
            print('F: {}'.format(cn))
            print('B: {}'.format(on))
            print('V: {}'.format(node.children))
            print('r: {}'.format(r))
    return rot, ds