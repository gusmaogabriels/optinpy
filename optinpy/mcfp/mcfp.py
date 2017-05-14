# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

def bigm(network,mcost,verbose=False):
    """
        big-M assumes non-negative flows
        .. network as  graph object
        .. mcost as numeric, the input cost of artificial arcs.
    """
    network.set_root_weight(mcost)         
    
    while True:
        non_basic = [arc for arc in network.non_basic]
        i = 1
        while len(non_basic)>0:
            print            
            i += 1
            arc = non_basic.pop(0)
            if arc.c_hat() <= 0:
                print(arc(),arc.c_hat())
                arc.pivot()
                break
            else:
                pass
        if len(non_basic) == 0:
            return(network)
        else:
            pass
        