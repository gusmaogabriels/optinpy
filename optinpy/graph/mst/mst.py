# -*- coding: utf-8 -*-

from ..base import argparser, graph 

def prim(network,verbose=False):
    '''
        Prims's algorithm
        This algorithm assumes undirected network inputs. If they aren't, it will assume so.
        ..network as optinpy.graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    arcs0 = list(network.arcs)
    costs0 = list(network.costs)
    on = list(nodes) # open nodes list (O)
    cn = []
    arcs = []
    t = 0
    while len(on)>0: 
        index = costs0.index(min(costs0))
        arc0 = arcs0.pop(index)
        costs0.remove(costs0[index])
        arcs += [arc0]
        print(arc0)
        if not all([i in cn for i in arc0]):
            cn += [on.pop(on.index(i)) for i in filter(lambda n : n not in cn, arc0)] # closed nodes list (F)
        else:
            pass # otherwise it would close a cycle
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('S: {}'.format(arcs))
            print('T: {}'.format(cn))
            print('V: {}'.format(on))
    return arcs
            
def kruskal(network,n0,verbose=False):
    '''
        Kruskal's algorithm
        ..network as optinpy.graph object
        ..n0 as integer node in the graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    arcs = list(network.arcs)
    rot = [n0 for i in range(0,len(nodes))] # route list
    on = list(nodes) # open nodes list (O)
    cn = [on.pop(on.index(n0))] # closed nodes list (F)
    r = n0
    t = 0
    while len(on)>0:
        r0 = r
        arcs = filter(lambda n : n[0] not in cn or n[1] not in cn, [[n,i] for n in cn for i in network.nodes[n].children])
        z = min(zip([network.costs[network.arcs.index(i)] for i in arcs],arcs))
        rot[nodes.index(z[1][1])] = z[1][0]
        r = z[1][1]
        arcs.remove(z[1])
        cn += [on.pop(on.index(r))]    
        if verbose:
            print('Iteration #{}'.format(t))
            print('r0: {}'.format(r0))
            print('r: {}'.format(r))
            print('S: {}'.format(arcs))
            print('T: {}'.format(cn))
            print('V: {}'.format(on))
    return rot
        
    