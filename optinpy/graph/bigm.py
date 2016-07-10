# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

from .base import node, argparser

def bigm(network,mcost,verbose=False):
    """
        big-M assumes non-negative flows
        .. network as  graph object
        .. mcost as numeric, the input cost of artificial arcs.
    """
    network.set_root_weight(mcost)    
    
    def loop_handler(source,destination): # find least common ancestor and retur the loop cycle
        nsource = [source.key]
        ndestination = [destination.key]
        asource = []
        adestination = []
        out_arc = [None,float('inf')]
        while not set(nsource) & set(ndestination):
            if source.__node_to_root__ != None:           
                nsource += [source.__node_to_root__.key]
                asource += [source.__arc_to_root__]
                if source.__arc_to_root__[1]>0 and source.__arc_to_root__[0].flow<out_arc[1]:
                    out_arc = [source.__arc_to_root__[0],source.__arc_to_root__[0].flow]                    
                else:
                    pass
                source = source.__node_to_root__
            else:
                pass
            if destination.__node_to_root__ != None:           
                ndestination += [destination.__node_to_root__.key]
                adestination += [destination.__arc_to_root__]
                if destination.__arc_to_root__[1]<0 and destination.__arc_to_root__[0].flow<out_arc[1]:
                    out_arc = [destination.__arc_to_root__[0],destination.__arc_to_root__[0].flow]                    
                else:
                    pass
                destination = destination.__node_to_root__
            else:
                pass
        lcn = list(set(nsource) & set(ndestination))[0]
        nsource = nsource[0:nsource.index(lcn)+1]
        ndestination = ndestination[0:ndestination.index(lcn)+1]
        asource = [asource[0:nsource.index(lcn)] if len(asource)>0 else []][0]
        adestination = [adestination[0:ndestination.index(lcn)] if len(adestination)>0 else[]][0]
        for a in asource:
            #print(a[0].source.key,a[0].destination.key,a[0].flow)
            #print('delta:',-out_arc[1]*a[1])
            a[0].flow -= out_arc[1]*a[1]
        for a in adestination:
            #print(a[0].source.key,a[0].destination.key,a[0].flow)
            #print('delta:',out_arc[1]*a[1])
            a[0].flow += out_arc[1]*a[1]
        out_arc[0].deactivate()
        return(out_arc[1])         
    
    while True:
        non_basic = [arc for arc in network.non_basic]
        while len(non_basic)>0:
            arc = non_basic.pop(0)
            if arc.cost - (arc.source.__pi__[0] - arc.destination.__pi__[0]) < 0:
                delta = loop_handler(arc.source,arc.destination)
                arc.activate()
                arc.flow += delta
                break
            else:
                pass
        if len(non_basic) == 0:
            return(network)
        else:
            pass
        