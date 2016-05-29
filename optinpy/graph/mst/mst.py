# -*- coding: utf-8 -*-

from ..base import argparser, graph 

def prim(network,n0,verbose=False):
    '''
        Prims's algorithm
        This algorithm assumes undirected network inputs. If they aren't, it will assume so.
        ..network as optinpy.graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    on = list(nodes) # open nodes list (O)
    arcs = []
    cn = [on.pop(on.index(n0))] # closed nodes list (F)
    r = n0
    t = 0
    while len(on)>0:
        r0 = r
        narcs = filter(lambda n : n[0] not in cn or n[1] not in cn,\
        [i for n in cn for j in [[[n,c] for c in network.nodes[n].children],\
        [[p,n] for p in network.nodes[n].parents]] for i in j])
        try:        
            z = min(zip([network.costs[network.arcs.index(i)] for i in narcs],narcs))
        except:
            raise Exception('Graph is disconnected.')
        arcs += [z[1]]
        r = filter(lambda n : n not in cn, z[1])[0]
        cn += [on.pop(on.index(r))]   
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('r0: {}'.format(r0))
            print('r: {}'.format(r))
            print('S: {}'.format(arcs))
            print('T: {}'.format(cn))
            print('V: {}'.format(on))
    return arcs
            
def kruskal(network,verbose=False):
    '''
        Kruskal's algorithm
        ..network as optinpy.graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    on = list(nodes) # open nodes list (O)
    cn = [] # closed nodes list
    costs = sorted(zip(network.costs,range(0,len(network.arcs)))) # zip list to link arc positions
    trees = []
    arcs = []
    t = 0
    while len(on)>0 or len(trees)>1:
        arc = list(network.arcs[costs[0][1]])
        if sum([arc[j] in i for j in [0,1] for i in trees])==2: # both nodes are within trees
            indexer = [sum([arc[j] in i for j in [0,1]]) for i in trees]            
            if not any([i==2 for i in indexer]): # this should link trees, for nodes are in different trees:
                arcs += [list(arc)]                
                pos = sorted([i[0] for i in filter(lambda x : x[1] != 0, zip(range(0,len(indexer)),indexer))])
                for i in trees[pos[1]]:
                    trees[pos[0]] += [i]
                trees.pop(pos[1])              
            else:
                pass # both nodes are already within a tree, nothing to be done.
        elif sum([arc[j] in i for j in [0,1] for i in trees])==1: # one of the nodes are within trees
            arcs += [list(arc)]
            indexer = filter(lambda k : k[1]>0, \
            zip(range(0,len(trees)),[sum([arc[j]*(j+1) in i for j in [0,1]]) for i in trees]))[0]
            trees[indexer[0]] += [arc[abs(indexer[1]-2)]]
            cn += [on.pop(on.index(arc[abs(indexer[1]-2)]))]
        else: # no nodes have been visited, thus build a new tree.
            arcs += [arc]  
            trees += [list(arc)]
            cn += [on.pop(on.index(i)) for i in list(arc)]
        costs.pop(0)
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('Trees: {}'.format(trees))
            print('S: {}'.format(arcs))
    return arcs
        
def boruvka(network,verbose=False):
    '''
        Boruvka's algorithm
        ..network as optinpy.graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    trees = [[i] for i in nodes]
    arcs0 = zip(range(0,len(network.arcs)),network.costs,network.arcs)
    arcs = []
    connections = [filter(lambda x : x!=None,[j if i in arcs0[j][2] else None for j in range(0,len(network.arcs))]) for i in nodes]
    t = 0
    while len(trees)>1:
        indexes = sorted([[i,sorted(zip([arcs0[j][1] for j in connections[i]],connections[i]))[0][1]] for i in range(0,len(trees))])
        i = 1
        j = 0
        while i<len(indexes):
            if indexes[i][1] == indexes[i-1][1]:
                connections[indexes[i-1][0]-j] = list(set(connections[indexes[i-1][0]-j]) ^ set(connections[indexes[i][0]-j]))
                trees[indexes[i-1][0]-j] += trees[indexes[i][0]-j]
                connections.pop(indexes[i][0]-j)
                trees.pop(indexes[i][0]-j)
                arcs += [arcs0[indexes[i][1]][2]]
                indexes.pop(i)
                j += 1
            else:
                i += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('Trees: {}'.format(trees))
            print('Arcs: {}'.format(arcs))
    return arcs
           