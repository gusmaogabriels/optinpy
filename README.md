**optinpy** 
==================================================================
*General linear and non-linear optimization methods in Python.*

**University of Campinas, UNICAMP**
	
. IA897 - Introdução à Otimização Matemática - *Introduction to Optimization* 
. IA881 - Otimização Linear - *Linear Programming*
. IA543 - Otimização Não Linear - *Nonlinear Optimization* (Prof. Takaaki)

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
      - [Golden-section](#golden-section-golden_section)
    - [Unconstrained optimization](#unconstrained-optimization-unconstrained)
      - [fminunc](#fminunc)
      - [parameters](#parameters-params)
        - [Gradient](#gradient-methodgradient)
        - [Newton](#newton-methodnewton)
        - [Modified Newton](#modified-newton-methodmodified-newton)
        - [Conjugate Gradient](#conjugate-gradient-methodconjugate-gradient)
		- [Quasi-Newton](#quasi-newton-methodquasi-newton)
			- [Davidon-Fletcher-Powell](#davidon-fletcher-powell-quasi-newtonhessian_updatedfp)
			- [Broyden-Fletcher-Goldfarb-Shanno](#broyden-fletcher-goldfarb-shanno-quasi-newtonhessian_updatebfgs)
			
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
Unidimensional minimizers that seek improving an objective function along a given descent direction, *d*, i.e. f(x) parametrized as f(α) = (x<sub>0</sub>+α×*d*).
Backtracking and 2nd-3rd order Interpolation methods enforde that the ensuing value satisfies Armijo's condition: f(x<sub>0</sub>+α×*d*) < f(x<sub>0</sub>)+*c*∇f(x<sub>0</sub>)'*d*, where *c* is an arbitrary threshold, *c* ∈ (0,1).
Unimodality and Golden-section are exploitation-based methods that successively probe and slice the domain determined by *d*, excluding, as of a quasi-convexity assumption on the α-domain, regions for which the function's value should be worse off.
#### Backtracking (`.backtracking`)
Finds an α value that satisfies Armijo's condition with successive value decrement by a factor ρ.
#### Interpolation, 2nd-3rd order (`.interp23`)
Finds an α value that satisfies Armijo's condition by successively minimizing a 2nd- or 3rd-order f(α) polynomial interpolated from the latest f-α pairs.
#### Unimodality (`.unimodality`)
Successive α-domain subsectioning by uniformly random probing.
#### Golden-section (`.golden_section`)
Successive α-domain subsectioning following the golden-ratio.

### Unconstrained Optimization (`.unconstrained`)
#### fminunc (`.fminunc`)
  [*fminunc*](#fminunc-fminunc) evokes the so far implemented unconstrained non-linear optimization algorithms given the parameters set.
  
#### parameters (`.params`)
  A dictionary object that holds the method/algorithm's set-up for the [*fminunc*](#fminunc-fminunc) function 
  
  **Gradient vs. Newton's Method, Modified-Newton** *(somewhere in between weighted by σ parameter)*, **and Conjugate Gradient** starting @ (2,2)<sup>*</sup>
  
  ![Alt Text](/raw/rosen.gif)
   
  **Log-scale error evolution**
  
  ![Alt Text](/raw/ErrorEvol.png)
  
  <sup>*</sup> Line-search method: 'interp23' with *alpha* = 1, *rho* = 0.5, *alpha_min* = 0.1, *c* = 0.1 (*Armijo's condition*); gradient and Hessian calculation from central algorithms and 10<sup>-6</sup> perturbation *epsilon*. *max_iter* = 10<sup>3</sup>
  
  **Standard parameters**
   ```python
   {'fminunc': # fminunc algorithm definition
       {'method': 'newton', # 'gradient', 'newton' or `'modified-newton'
        'params': 
           {'gradient':
                {'max_iter':1e3},
            'newton':
                {'max_iter':1e3},
            'modified-newton':
                {'sigma' : 1, 'max_iter':1e3}, # sigma is the lower bound for the modified Hessian eigenvalue
            'conjugate-gradient':
                {'max_iter':1e3},    
          }},
    'jacobian': # jacobian algorithm definition
       {'algorithm':'central','epsilon':1e-6}, # algorithm = 'central', 'forward', 'backward'; epsilon = perturbation
    'hessian':
       {'algorithm':'central','epsilon':1e-6}, # algorithm = 'central', 'forward', 'backward'; epsilon = perturbation
    'linesearch':
       {'method':'backtracking', # 'backtracking', 'interp23, 'unimodality' or 'golden-section'
        'params':
           {'backtracking':
                {'alpha':1,'rho':0.5,'c':1e-4,'max_iter':1e3}, # alpha = initial step scale; rho = step reduction factor; c = Armijo's parameter
            'interp23':
                {'alpha':1,'alpha_min':0.1,'rho':0.5,'c':1e-4,'max_iter':1e3}, # alpha_min = minimum ultimate alpha below which 3rd order interpolation ensues
            'unimodality':
                {'b':1,'threshold':1e-4,'max_iter':1e3}, # b = initial step scale in derivatives domain; threshold: variation threshold
            'golden_ratio':
                {'b':1,'threshold':1e-4,'max_iter':1e3}
           }
       }
   }   
   ```
    
  - ##### Gradient (`method='gradient'`)
    The gradient algorithm minimizes f(x) by first-order approximation: f(x) = f(x<sub>0</sub>) + ∇f(x<sub>0</sub>)'Δx, with descent direction, *d*, given by:
    
      *d* = -∇f(x<sub>0</sub>).
      
    The ultimate iteration step is given by the minimization of f(α) = f(x<sub>0</sub>) - α∇f(x<sub>0</sub>) with a lineserach substep.
    
  - ##### Newton (`method='newton'`)
    Newton's method minimizes f(x) as of a second order approximation of the function f(x) = f(x<sub>0</sub>) + ∇f(x<sub>0</sub>)'Δx + Δx'H(x<sub>0</sub>)Δx, where H(x<sub>0</sub>) is the Hessian matrix.
    The descent direction is given by: 
    
      *d* = -H(x<sub>0</sub>)<sup>-1</sup>\*∇f(x<sub>0</sub>).
    
  - ##### Modified Newton (`method='modified-newton'`)
    The modified-Newton's algorithm handles the inversion of H(x<sub>0</sub>) by enforcing positive eigenvalues so that Cholesky's decomposition can be used to solve H(x<sub>0</sub>)\**d* = -∇f(x<sub>0</sub>) as a system of lower-triangular matrices: 
    
      H(x) = L\*L'
      
      L\*y = -∇f(x<sub>0</sub>) 
      
      L'\**d* = y.

  - ##### Conjugate Gradient (`method='conjugate-gradient'`)
    The conjugate gradient algorithm builds a set of Hessian-orthogonal (*Q*-orthogonal) directions as of Gram-Schmidt *Q*-orthogonalization so that descent directions, *d*, are *Q*-orthogonal and preceding gradients are orthogonal: *d*<sup>k+1</sup> = *p*<sup>k+1</sup> - ∑<sup>k</sup><sub>i=0</sub>(*p*<sup>k+1</sup>'*Qd*<sup>i</sup>/*d*<sup>i</sup>'*Q**d*<sup>i</sup>)*d*<sup>i</sup>, where *p* is a linear independent set. Replacing *p*<sup>i</sup> by -∇f(x<sub>i</sub>), leads us to:
    
      *d*<sup>k+1</sup> = -∇f(x<sub>k+1</sub>) + ∇f(x<sub>k+1</sub>)'H(x<sub>k+1</sub>)*d*<sup>k</sup>/*d*<sup>k</sup>'H(x<sub>k+1</sub>)*d*<sup>k</sup>

	  
  - ##### Quasi-Newton (`method='quasi-newton'`)
    Quasi-Newton methods are in between gradient and Newton's method, with successive approximations of the Hessian and its inverse matrix.
	
	for the following updates, consider:
	
	  q<sup>k</sup> = ∇f(x<sub>k+1</sub>) - ∇f(x<sub>k</sub>) 
	  
	  p<sup>k</sup> = α<sub>k</sub>d<sup>k</sup>
    
	- ###### Davidon-Fletcher-Powell ('quasi-newton':'hessian_update':'dfp')
      
		Rank-2 correction of the inverse Hessian as of a sum of 2 rank-1 matrices.
			
		*H*<sup>-1</sup><sub>k+1</sub> = *H*<sup>-1</sup><sub>k</sub> + p<sup>k</sup>(p<sup>k</sup>)'/(p<sup>k</sup>)'q<sup>k</sup> - *H*<sup>-1</sup><sub>k</sub>q<sup>k</sup>(q<sup>k</sup>)'*H*<sup>-1</sup><sub>k</sub>/(q<sup>k</sup>)'*H*<sup>-1</sup><sub>k</sub>q<sup>k</sup>

	- ###### Broyden-Fletcher-Goldfarb-Shanno ('quasi-newton':'hessian_update':'BFGS')
		
		Also a Rank-2 correction of the inverse Hessian as of a sum of 2 rank-1 matrices.
			
		*H*<sup>-1</sup><sub>k+1</sub> = *H*<sup>-1</sup><sub>k</sub> + (1+(q<sup>k</sup>)'*H*<sup>-1</sup><sub>k</sub>q<sup>k</sup>/(q<sup>k</sup>)'p<sup>k</sup>)p<sup>k</sup>(p<sup>k</sup>)'/(p<sup>k</sup>)'q<sup>k</sup> - (p<sup>k</sup>(q<sup>k</sup>)'*H*<sup>-1</sup><sub>k</sub>+*H*<sup>-1</sup><sub>k</sub>q<sup>k</sup>(p<sup>k</sup>)')/(q<sup>k</sup>)'p<sup>k</sup>

	
Copyright © 2016 - Gabriel Sabença Gusmão

[![linkedin](https://static.licdn.com/scds/common/u/img/webpromo/btn_viewmy_160x25.png)](https://br.linkedin.com/pub/gabriel-saben%C3%A7a-gusm%C3%A3o/115/aa6/aa8)

[![researchgate](https://www.researchgate.net/images/public/profile_share_badge.png)](https://www.researchgate.net/profile/Gabriel_Gusmao?cp=shp)
