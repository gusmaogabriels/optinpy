

**optinpy** 
==================================================================
*General linear and non-linear optimization methods in Python.*

*pip install* -> `pip install --upgrade https://github.com/gusmaogabriels/optinpy/zipball/master`

## Table of Content
  - [**Linear Programming**](#linear-programming)
    - [Graphs](#graphs-graph)
    - [Minimum-cost flow problem](#minimum-cost-flow-problem-mcfp)
    - [Shortest-path Algorithms](#shortest-path-algorithms-sp)
    - [Minimum spanning-tree Algorithms](#minimum-spanning-tree-algorithms-mst)    
  - [**Non-linear Optimization**](#non-linear-optimization)
    - [Line-search](#line-search-linesearch)
      - [Backtracking](#backtracking-backtracking)
      - [Interpolation 2nd-3rd](#interpolation-2nd-3rd-order-interp23)
      - [Unimodality](#unimodality-unimodality)
      - [Golden-ratio](#golden-ratio-golden_ratio)
    - [Unconstrained-optimization](#Unconstrained-optimization-unconstrained)
      - [fminunc](#fminunc)
      - [parameters](#parameters-params)
        - [Gradient](#gradient-methodgradient)
        - [Newton](#newton-methodnewton)
        - [Modified Newton](#modified-newton-methodmodified-newton)

## **Linear Programming**
### Graphs (`.graph`)
  - Equivalent *tableau* (`.tableau`)   -> *under development*
  - Nodes objetcs (`.node`)
  - Arcs objects (`.arc`)
  
### Minimum-cost flow problem (`.mcfp`)
   - Big-M (`.bigm`)                   
        
    **Example**
    ```python
    ### BIG-M ###
    n3 = optinpy.graph() # Create an graph object
    bs = zip(range(1,6),[10,4,0,-6,-8]) # Set node net production (+) or consumption (-)
    for b in bs:
        n3.add_node(*b) # Add node to graph  and assign them a net production (+) or consumption (-)
    connections = [[1,2,1],[1,3,8],[1,4,1],[2,3,2],[3,4,1],[3,5,4],[4,5,12],[5,2,7]] # Define arcs [from,to,cost]
    for c in connections:
        n3.add_connection(*c) # Add arcs (connections) to the graph
    optinpy.bigm(n3,20) # MCF problem via Big-M
    ```
 
### Shortest-path Algorithms (`.sp`)
  - Dijkstra's (`.dijkstra`)
  - Ford-Bellman-Moore's (`.fmb`)
     
     **Example**
     ```python
     ### Exampe 1 ###
     n = optinpy.graph()  # Create an graph object
     connections = [[1,2,2],[1,3,4],[1,4,5],[2,4,2],[3,4,1]]
     for c in connections:
         n.add_connection(*c) # Define arcs [from,to,cost]
     rot,ds = optinpy.sp.fmb(n,1,verbose=True) # Shortest-path via Ford-Belman-Moore   
     rot,ds = optinpy.sp.dijkstra(n,1,verbose=True) # Shortest-path via Dijkstra

     ### Exampe 2 ### https://www.youtube.com/watch?v=obWXjtg0L64
     n = optinpy.graph()  # Create an graph object
     connections = [[1,2,10],[1,6,8],[2,4,2],[3,2,1],[4,3,-2],[5,2,-4],\
                     [5,4,-1],[6,5,1]] # Define arcs [from,to,cost]
     for c in connections:
         n.add_connection(*c) # Add arcs (connections) to the graph
     rot,ds = optinpy.sp.fmb(n,1,verbose=True) # Shortest-path via Ford-Belman-Moore
     
     ### Exampe 3 ### https://www.youtube.com/watch?v=gdmfOwyQlcI
     n = optinpy.graph()  # Create an graph object
     connections = [['A','B',4],['A','E',7],['A','C',3],['B','C',6],['B','D',5],['C','D',11],\
                     ['C','E',8],['D','E',2],['D','F',2],['D','G',10],['E','G',5]] # Define arcs [from,to,cost]
     for c in connections:
         n.add_connection(*c) # Add arcs (connections) to the graph
     rot,ds  = optinpy.sp.dijkstra(n,'A',verbose=True) # Shortest-path via Dijkstra
     ```  
  
### Minimum spanning-tree Algorithms (`.mst`)
  - Prim's (`.prim`)
  - Kruskal's alike (see commit notes) (`.kruskal`)
  - Borůvka's (`.boruvka`)
  
     **Example**
     ```python
     n = optinpy.graph()  # Create an graph object
     connections = [['A','B',7],['A','D',5],['B','C',8],['B','D',9],['B','E',7],['C','E',5],\
                ['D','E',15],['D','F',6],['E','F',8],['E','G',9],['F','G',11]] # Define arcs [from,to,cost]
     for c in connections:
         n.add_connection(*c) # Add arcs (connections) to the graph
     n2 = optinpy.graph()
     connections = [['A','B',13],['A','C',6],['B','C',7],['B','D',1],['C','E',8],['C','D',14],\
                    ['C','H',20],['D','E',9],['D','F',3],['E','F',2],['E','J',18],['G','H',15],\
                    ['G','I',5],['G','J',19],['G','K',10],['H','J',17],['I','K',11],['J','K',16],\
                    ['J','L',4],['K','L',12]] # Define arcs [from,to,cost]
     for c in connections:
         n2.add_connection(*c) # Add arcs (connections) to the graph
     # Assessing n2
     print("Prims's")
     arcs1 = optinpy.mst.prim(n2,'A',verbose=True) # Minimum spanning-tree via Prim
     print("Kruskal's")
     arcs2 = optinpy.mst.kruskal(n2,verbose=True) # Minimum spanning-tree via Kruskal
     print("Boruvka's")
     arcs3 = optinpy.mst.boruvka(n2,verbose=True) # Minimum spanning-tree via Boruvka
     ```
## **Non-linear Optimization**
### Line-search (`.linesearch`)
#### Backtracking (`.backtracking`)
#### Interpolation, 2nd-3rd order (`.interp23`)
#### Unimodality (`.unimodality`)
#### Golden-ratio (`.golden_ratio`)

### Unconstrained Optimization (`.unconstrained`)
#### fminunc (`.fminunc`)
  [*fminunc*](#fminunc-fminunc) evokes the so far implemented unconstrained non-linear optimization algorithms given the parameters set.
  
#### parameters (`.params`)
  A dictionary object that holds the method/algorithm definition for the [*fminunc*](#fminunc-fminunc) function
  
  ![Alt Text](/raw/rosen.gif)
  
  **Standard parameters**
   ```python
   {'fminunc': # fminunc algorithm definition
       {'method': 'newton', # 'gradient', 'newton' or `'modified-newton'
        'params': 
           {'gradient':
                 {},
            'newton':
                {},
            'modified-newton':
                {'sigma' : 1} # lower bound for the modified Hessian eigenvalue
          }},
    'jacobian': # jacobian algorithm definition
       {'algorithm':'central','epsilon':1e-6}, # algorithm = 'central', 'forward', 'backward'l epsilon = perturbation
    'hessian':
       {'algorithm':'central','epsilon':1e-6}, # algorithm = 'central', 'forward', 'backward'l epsilon = perturbation
    'linesearch':
       {'method':'backtracking', # 'backtracking', 'interp23, 'unimodality' or 'golden_ratio'
        'params':
           {'backtracking':
                {'alpha':1,'rho':0.5,'c':1e-4}, # alpha = initial step scale; rho = step reduction factor; c = Armijo's parameter
            'interp23':
                {'alpha':1,'alpha_min':0.1,'rho':0.5,'c':1e-4}, # alpha_min = minimum ultimate alpha below which 3rd order interpolation ensues
            'unimodality':
                {'b':1,'threshold':1e-4}, # b = initial step scale in derivatives domain; threshold: variation threshold
            'golden_ratio':
                {'b':1,'threshold':1e-4}
           }
       }
   }   
   ```
    
  - ##### Gradient (`method='gradient'`)
    The gradient algorithm minimizes f(x) by first-order approximation: f(x) = f(x<sub>0</sub>) + ∇f(x<sub>0</sub>)'Δx, with descent direction, *d*, given by:
    
      *d* = -∇f(x<sub>0</sub>).
      
    The ultimate iteration step is given by the minimization of f(α) = f(x<sub>0</sub>) - α∇f(x<sub>0</sub>) with a lineserach algorithm.
    
  - ##### Newton (`method='newton'`)
    Newton's method minimizes f(x) as of a second order approximation of the function f(x) = f(x<sub>0</sub>) + ∇f(x<sub>0</sub>)'Δx + Δx'H(x<sub>0</sub>)Δx, where H(x<sub>0</sub>) is the Hessian matrix.
    The descent direction is given by: 
    
      *d* = -Hx<sub>0</sub><sup>-1</sup>\*∇f(x<sub>0</sub>).
    
  - ##### Modified Newton (`method='modified-newton'`)
    The modified-Newton's algorithm handles the inversion of H(x<sub>0</sub>) by enforcing positive eigenvalues so that Cholesky's factorization can be used to solve H(x<sub>0</sub>)\**d* = -∇f(x<sub>0</sub>) as a system of lower-triangular matrices: 
    
      H(x) = L\*L'
      
      L\*y = -∇f(x<sub>0</sub>) 
      
      L'\**d* = y.

Copyright © 2016 - Gabriel Sabença Gusmão

[![linkedin](https://static.licdn.com/scds/common/u/img/webpromo/btn_viewmy_160x25.png)](https://br.linkedin.com/pub/gabriel-saben%C3%A7a-gusm%C3%A3o/115/aa6/aa8)

[![researchgate](https://www.researchgate.net/images/public/profile_share_badge.png)](https://www.researchgate.net/profile/Gabriel_Gusmao?cp=shp)
