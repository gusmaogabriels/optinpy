# -*- coding: utf-8 -*-

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
    while len(on)>1:
        r0 = r
        z = float('inf')
        narc = []
        for n in [network.nodes[i] for i in cn]:
            for arc in filter(lambda x : x[0] not in cn or x[1] not in cn, n.get_connections()):
                narc, z = [[narc,z] if network.arcs[arc[0]][arc[1]].cost>z else [arc,network.arcs[arc[0]][arc[1]].cost]][0]    
        arcs += [narc]
        r = filter(lambda n : n not in cn, arcs[-1])[0]
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
    costs = sorted([[network.arcs[f][t].cost,[f,t]] for f in network.arcs.keys() for t in network.arcs[f].keys()])# list to link arc positions
    trees = []
    arcs = []
    t = 0
    while len(on)>1 or len(trees)>1:
        arc = costs[0][1]
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
    nodes_map = [[i] for i in range(0,len(network.nodes)-1)]
    nodes = dict(zip(list(set(network.nodes.keys())^set([0])),nodes_map))
    trees = [[i] for i in nodes.keys()]
    arcs0 = [[network.arcs[j][i].cost,[j,i]] for j in network.arcs.keys() for i in network.arcs[j].keys()]
    arcs0 = [[i,arcs0[i][0],arcs0[i][1]] for i in range(0,len(arcs0))]  
    arcs = []
    connections = [filter(lambda x : x!=None,[j if i in arcs0[j][2] else None for j in range(0,len(arcs0))]) for i in nodes.keys()]
    t = 0
    while len(trees)>1:
        indexes = [[[filter(lambda x : x in trees[i], j)[0],filter(lambda x : x not in trees[i], j)[0],j] for j in [arcs0[sorted(zip([arcs0[j][1] for j in connections[i]],connections[i]))[0][1]][2]]][0] \
        for i in range(0,len(trees))]
        while len(indexes)>0 and len(trees)>1:
            nto, nfrom  = sorted(zip([nodes[indexes[0][0]][0],nodes[indexes[0][1]][0]],[0,1]))             
            connections[nto[0]] = list(set(connections[nto[0]]) ^ set(connections[nfrom[0]]))
            connections.pop(nfrom[0])
            trees[nto[0]] += trees[nfrom[0]]
            trees.pop(nfrom[0])
            nodes[indexes[0][nfrom[1]]][0] = nto[0]
            for i in nodes_map:
                if i[0] >= max([nto[0],nfrom[0]]):
                    i[0] -= 1
                else:
                    pass
            arcs += [indexes[0][2]]
            indexes.pop(0)
            while len(indexes)>0 and indexes[0][2] in arcs:
                indexes.pop(0)
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('Trees: {}'.format(trees))
            print('Arcs: {}'.format(arcs))
    return arcs
           