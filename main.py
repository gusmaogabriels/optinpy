# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:24:08 2016

@author: GABRIS46
"""

import optinpy
n = optinpy.graph()
connections = [[1,2,1],[1,3,3],[2,3,1],[2,4,2],[2,5,3],[3,5,2],[4,5,-3],[4,6,3],[5,6,2]]
for c in connections:
    n.add_connection(*c)
optinpy.mst.boruvka(n,verbose=True)\

