# -*- coding: utf-8 -*-

from ..base import argparser, graph 

def fmb(network,n0,verbose=False):
    '''
        Standard Ford-Moore-Bellman's algorithm
        ..network as optinpy.graph object
        ..n0 as integer node in the graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    arcs = network.arcs
    ds = [float('inf') for i in range(0,len(nodes))] # distances list
    rot = [n0 for i in range(0,len(nodes))] # route list
    ds[nodes.index(n0)] = 0
    t = 0
    for i in range(0,len(nodes)):
        ds0 = list(ds)
        for j in range(0,len(arcs)):
            if ds[nodes.index(arcs[j][1])] > network.costs[j] + ds[nodes.index(arcs[j][0])]:
                if i == len(nodes)-1: # this needs validation
                    print('Negative-weight cycle exists!')
                    return rot, ds
                ds[nodes.index(arcs[j][1])] = network.costs[j] + ds[nodes.index(arcs[j][0])]
                rot[nodes.index(arcs[j][1])] = arcs[j][0]
            else:
                pass
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('d #{}'.format(ds))
        if ds == ds0:
            return rot, ds
        else:
            pass
    return rot, ds

def dijkstra(network,n0,verbose=False):
    '''
        Standard Dijkstra's algorithm
        ..network as optinpy.graph object
        ..n0 as integer node in the graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    arcs = network.arcs
    ds = [float('inf') for i in range(0,len(nodes))] # distances list
    rot = [n0 for i in range(0,len(nodes))] # route list
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