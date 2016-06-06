# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:24:08 2016

@author: GABRIS46
 """

import optinpy
n = optinpy.graph()
connections = [['A','B',7],['A','D',5],['B','C',8],['B','D',9],['B','E',7],['C','E',5],\
                ['D','E',15],['D','F',6],['E','F',8],['E','G',9],['F','G',11]]
for c in connections:
    n.add_connection(*c)
print("Prims's")
arcs1 = optinpy.mst.prim(n,'A',verbose=True)
n2 = optinpy.graph()
connections = [['A','B',13],['A','C',6],['B','C',7],['B','D',1],['C','E',8],['C','D',14],\
                ['C','H',20],['D','E',9],['D','F',3],['E','F',2],['E','J',18],['G','H',15],\
                ['G','I',5],['G','J',19],['G','K',10],['H','J',17],['I','K',11],['J','K',16],\
                ['J','L',4],['K','L',12]]
for c in connections:
    n2.add_connection(*c)
print("Kruskal's")
arcs2 = optinpy.mst.kruskal(n,verbose=True)
print("Boruvka's")
arcs3 = optinpy.mst.boruvka(n,verbose=True)


""" SHORTEST PATH
n = optinpy.graph()
connections = [[1,2,2],[1,3,4],[1,4,5],[2,4,2],[3,4,1]]
for c in connections:
    n.add_connection(*c)
rot,ds = optinpy.sp.dijkstra(n,1,verbose=True)
rot,ds = optinpy.sp.fmb(n,1,verbose=True)

connections = [['A','B',7],['A','D',5],['B','C',8],['B','D',9],['B','E',7],['C','E',5],\
                ['D','E',15],['D','F',6],['E','F',8],['E','G',9],['F','G',11]]
for c in connections:
    n.add_connection(*c)
rot,ds  = optinpy.sp.dijkstra(n,'A',verbose=True)
rot,ds = optinpy.sp.fmb(n,'A',verbose=True)
"""