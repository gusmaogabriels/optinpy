# -*- coding: utf-8 -*-

def fmb(network,n0,verbose=False):
    '''
        Standard Ford-Moore-Bellman's algorithm
        ..network as optinpy.graph object
        ..n0 as integer node in the graph object
        ..verbose as boolean
    '''
    nodes = network.nodes.keys()
    ds = dict(zip(nodes,[float('inf') for i in range(0,len(nodes))])) # distances list
    rot = dict(zip(nodes,[n0 for i in range(0,len(nodes))])) # route list
    ds[n0] = 0
    t = 0
    for i in range(0,len(nodes)):
        ds0 = list(ds)
        for j in network.arcs.keys():
            for k in network.arcs[j].keys():
                if ds[network.arcs[j][k].destination.key] > network.arcs[j][k].cost + ds[network.arcs[j][k].source.key]:
                    ds[network.arcs[j][k].destination.key] = network.arcs[j][k].cost + ds[network.arcs[j][k].source.key]
                    rot[network.arcs[j][k].destination.key] = ds[network.arcs[j][k].source.key]
                else:
                    pass
        if i == len(nodes): # this needs validation
            print('Negative-weight cycle exists!')
            return rot, ds
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('d: {}'.format(ds))
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
    ds = dict(zip(nodes,[float('inf') for i in range(0,len(nodes))])) # distances list
    rot = dict(zip(nodes,[n0 for i in range(0,len(nodes))])) # route list
    on = list(nodes) # open nodes list (O)
    cn = [on.pop(on.index(n0))] # closed nodes list (F)
    ds[n0] = 0
    r = n0
    t = 0
    while len(on)>0:
        r0 = r
        node = network.nodes[r]
        for i in [i.key for i in node.__children__]:
            if ds[i] > network.arcs[r][i].cost + ds[r]:
                ds[i] = network.arcs[r][i].cost + ds[r]
                rot[i] = r
            else:
                pass
        r = min(zip([ds[i] for i in on], on))[1]
        cn += [on.pop(on.index(r))]
        t += 1
        if verbose:
            print('Iteration #{}'.format(t))
            print('r0: {}'.format(r0))
            print('d: {}'.format(ds))
            print('rot: {}'.format(rot))
            print('F: {}'.format(cn))
            print('B: {}'.format(on))
            print('V: {}'.format([i.key for i in node.__children__]))
            print('r: {}'.format(r))
    return rot, ds