**optinpy**
==================================================================
*General linear and non-linear optimization methods in Python.*

*pip install* -> `pip install --upgrade https://github.com/gusmaogabriels/optinpy/zipball/master`

**The University of Campinas, UNICAMP**

* IA897 - Introdução à Otimização Matemática - *Introduction to Optimization*
* IA881 - Otimização Linear - *Linear Programming*
* IA543 - Otimização Não Linear - *Nonlinear Optimization* (Prof. Takaaki)


## Table of Content
  - [**Linear Programming**](#linear-programming)
    - [Graphs](#graphs-graph)
    - [Minimum-cost flow problem](#minimum-cost-flow-problem-mcfp)
    - [Shortest-path Algorithms](#shortest-path-algorithms-sp)
    - [Minimum spanning-tree Algorithms](#minimum-spanning-tree-algorithms-mst)    
    - [Simplex **BEING FIXED**](#simplex-simplex)
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
			- [Davidon-Fletcher-Powell (DFP)](#davidon-fletcher-powell-quasi-newtonhessian_updatedfp)
			- [Broyden-Fletcher-Goldfarb-Shanno (BFGS)](#broyden-fletcher-goldfarb-shanno-quasi-newtonhessian_updatebfgs)
    - [Constrained optimization](#constrained-optimization-constrained)
      - [fmincon](#fmincon)
      - [parameters](#parameters-params)
        - [Projected Gradient](#projected-gradient-methodprojected-gradient)
  - [**Numerical Differentiation**](#numerical-differentiation)
    - [Finite Difference](#finite-difference-finitediff)
      - [Jacobian](#jacobian-jacobian)
      - [Hessian](#hessian-hessian)

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
### Simplex (`.simplex`)
**BEING FIXED: AFTER ADDITION OF LB/UB DEFINITION FOR *x*, GLITCHES CAME ABOUT... IT SHOULD BE FIXED ASAP**
  - Base Class Constructor (`S = optinpy.simplex(A,b,c,lb,ub)`)

  	Define a linear optimization problem *S*: `S = optinpy.simplex(A,b,c,lb,ub)` to find the *minimum* value of *c*×*x* (default) subject  to *A*×*x* ≤ *b*, where *A* is a *n*×*m* matrix holding the constraints coefficients, *b* ∈ R<sup>*n*</sup> and *c* ∈ R<sup>*m*</sup> is the objective function cofficients, *lb* and *ub* are the lower and upper bound values in R<sup>*n*</sup> for *x*, respectively.

  - Primal Step (`S.primal()`)

  	Given *S* a Simplex object, `S.primal()` carries out a primal pivotting as of a primal feasible *x*, including in the basis set the basis of greatest improvement of the Objective Functionl, while still respecting the primal feasibility*A*×*x* ≤ *b*, then determining which basis should leave the basis set.

  - Dual Simplex (`S.dual()`)

  	Given *S* a Simplex object, `S.dual()` carries out a dual pivotting as of a dual feasible *x*, so that there is first defined which basis should leave the basis set and the which one should enter while keeping the dual feasibility.

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

  <sup>*</sup> Line-search method: 'interp23' with *alpha* = 1, *rho* = 0.5, *alpha_min* = 0.1, *c* = 0.1 (*Wolfe's condition*); gradient and Hessian calculation from central algorithms and 10<sup>-6</sup> perturbation *epsilon*. *max_iter* = 10<sup>3</sup>

  **Standard parameters**
   ```python
   {'fminunc': # fminunc algorithm definition
       {'method': 'newton', # 'gradient', 'newton', `'modified-newton', 'fletcher-reeves' or 'quasi-newton'
        'params':
           {'gradient':
                {'max_iter':1e3},
            'newton':
                {'max_iter':1e3},
            'modified-newton':
                {'sigma' : 1, 'max_iter':1e3}, # sigma is the lower bound for the modified Hessian eigenvalue
            'conjugate-gradient':
                {'max_iter':1e3},
            'fletcher-reeves':
                {'max_iter':1e3},
            'quasi-newton':
                {'max_iter':1e3,'hessian_update':'davidon-fletcher-powell'} # hessian_update is either dfp or BFGS
           }},
    'jacobian': # jacobian algorithm definition
       {'algorithm':'central','epsilon':sqrt(eps)}, # algorithm = 'central', 'forward', 'backward'; epsilon = perturbation
    'hessian':
       {'algorithm':'central','epsilon':sqrt(eps),'initial':None}, # algorithm = 'central', 'forward', 'backward'; epsilon = perturbation, 'inital' = initial hessian with (size of x)-by-(size of x) or None for identity.
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

### Constrained Optimization (`.constrained`)
#### fmincon (`.fmincon`)
  [*fmincon*](#fmincon-fmincon) extends [*fminunc*](#fminunc-fminunc) functionality for cases in which linear and nonlinear constraints are in play.

#### parameters (`.params`)
  A dictionary object that holds the method/algorithm's set-up for the [*fmincon*](#fmincon-fmincon-fmincon) function

  **Projected-gradient algorithm lifting off from the axis center @ (0,0) to a feasible starting point by Simplex with -∇f(x<sub>0</sub>) cost**

  ![Alt Text](/raw/rosen_proj.gif)

  Constraints:
  
  2x<sub>1</sub>  -  <sub>2</sub>  ≤ -1  
  5x<sub>1</sub>  + 3x<sub>2</sub> ≤  0  
  2x<sub>1</sub>  +  x<sub>2</sub> ≤  3  
   x<sub>1</sub>  - 2x<sub>2</sub> ≤  2
  
  <sup>*</sup> Line-search method: 'interp23' with *alpha* = 1, *rho* = 0.6, *alpha_min* = 0.1, *c* = 0.1 (*Wolfe's condition*); gradient and Hessian calculation from central algorithms and *eps*<sup>0.5</sup> perturbation *epsilon*, where *eps* stands for the smallest *float64* number suchs that 1.0 + *eps* != 0. *max_iter* = 10<sup>3</sup>

  **Standard parameters**
   ```python
   {'fmincon': # fmincon algorithm definition
       {'method': 'projected-gradient', # 'projected-gradient' is so far the only option (**reduced-gradient under delopment**)
        'params':
           {'gradient':
                {'max_iter':1e3},
            'newton':
                {'max_iter':1e3},
            'modified-newton':
                {'sigma' : 1, 'max_iter':1e3}, # sigma is the lower bound for the modified Hessian eigenvalue
            'conjugate-gradient':
                {'max_iter':1e3},
            'fletcher-reeves':
                {'max_iter':1e3},
            'quasi-newton':
                {'max_iter':1e3,'hessian_update':'davidon-fletcher-powell'} # hessian_update is either dfp or BFGS
           }},
    'jacobian': # jacobian algorithm definition
       {'algorithm':'central','epsilon':sqrt(eps)}, # algorithm = 'central', 'forward', 'backward'; epsilon = perturbation
    'hessian':
       {'algorithm':'central','epsilon':sqrt(eps),'initial':None}, # algorithm = 'central', 'forward', 'backward'; epsilon = perturbation, 'inital' = initial hessian with (size of x)-by-(size of x) or None for identity.
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

  - ##### Projected-gradient (`method='projected-gradient'`)
    The projected-gradient algorithm minimizes f(x) by first-order approximation: f(x) = f(x<sub>0</sub>) + ∇f(x<sub>0</sub>)'Δx, with descent direction, *d* under inequalities constraints *A*<=*b* and equalities constraints *A*<sub>eq</sub>=*b*<sub>eq</sub> given by:

    *d* = -*P*∇f(x<sub>k</sub>).

    where *P* is the orthogonal projection matrix on the *nullspace* of the subspace defined by the active constraints.

    *P* is given by *I* - *A*<sub>k</sub>'(*A*<sub>k</sub>*A*<sub>k</sub>')<sup>-1</sup>*A*<sub>k</sub> and *I* is the identity matrix.

    *A*<sub>k</sub> is *n*-by-*m* consists of the union of the set of active inequality constraints and the equality constrints set.

    The ultimate iteration step is given by the minimization of f(α) = f(x<sub>0</sub>) - α*d*(x<sub>k</sub>) with a lineserach substep.

    If *d* is less than a given arbitrary tolerance, the Lagrangian multipliers are computed to ckeck whether the Karush-Kuhn-Tucker (KKT) conditions are satisfied. If any inequality constraint lagragian multiplier is less than zero, the constraint is relaxed, the respective line in *A*<sub>k</sub> is removed and a the iterative process proceeds with *A*<sub>k+1</sub>. Otherwise the solution satisfies the KKT conditions and the point is said to be optimal.

    The lagrangian multipliers are given by -(*A*<sub>k</sub>*A*<sub>k</sub>')<sup>-1</sup>*A*<sub>k</sub>∇f(x<sub>k</sub>).
    
    **Example**: f(*x*) = ||*x*||²-2*x*<sub>1</sub>-3*x*<sub>4</sub>, begining @ (0,0,0,0) to a feasible starting point by Simplex with -∇f(x<sub>0</sub>) cost
    <sup>*</sup> Line-search method: 'interp23' with *alpha* = 1, *rho* = 0.6, *alpha_min* = 0.1, *c* = 0.1 (*Wolfe's condition*); gradient and Hessian calculation from central algorithms and *eps*<sup>0.5</sup> perturbation *epsilon*, where *eps* stands for the smallest *float64* number suchs that 1.0 + *eps* != 0. *max_iter* = 10<sup>3</sup>  
    
## **Numerical Differentiation**
### Finite Difference (`.finitediff`)
Numerical routines to estimate the *Jacobian* and *Hessian* matrices of a function at a given point, *x*<sub>0</sub>, as of small perturbations along it. As default, the perturbation ε is taken as the square root of the machine precision for a floating point type, *eps*, such that: 1.0 + *eps* != 0, and the algorithm for both *Jacobian* and *Hessian* is the central algorithm.

- ##### Jacobian (`.jacobian`)
  Typically, the first derivatives of a function or array of functions may be estimated by three different first-order approximation formulas/algorithms:
  
  - Central:
  
    ∇<sub>x<sub>j</sub></sub>f<sub>i</sub> = (f<sub>i</sub>(*x*<sub>j,0</sub>+ε)-f<sub>i</sub>(*x*<sub>j,0</sub>-ε))/2ε
  
  - Forward:
  
     ∇<sub>x<sub>j</sub></sub>f<sub>i</sub> = (f<sub>i</sub>(*x*<sub>j,0</sub>+ε)-f<sub>i</sub>(*x*<sub>j,0</sub>))/ε
  
  - Backward:
  
     ∇<sub>x<sub>j</sub></sub>f<sub>i</sub> = (f<sub>i</sub>(*x*<sub>j,0</sub>)-f<sub>i</sub>(*x*<sub>j,0</sub>-ε))/ε
  
- ##### Hessian (`.hessian`)
  The following alogirthms are used for the estimation of the second derivatives for a function from a reference point *x*<sub>0</sub>:
  
  - Central:
  
     On-diagonal terms: 𝛿²f/𝛿x<sub>j</sub>² = (-f(x<sub>j,0</sub>+2ε)+16f(x<sub>j,0</sub>+ε)-30f(x<sub>j,0</sub>)+16f(x<sub>j,0</sub>-ε)-f(x<sub>j,0</sub>-2ε))/12ε²
     
     Off-diagonal terms: 𝛿²f/𝛿x<sub>j</sub>𝛿x<sub>k</sub> = (f(x<sub>j,0</sub>+ε,x<sub>k,0</sub>+ε)-f(x<sub>j,0</sub>+ε,x<sub>k,0</sub>-ε)-f(x<sub>j,0</sub>-ε,x<sub>k,0</sub>+ε)+f(x<sub>j,0</sub>-ε,x<sub>k,0</sub>-2))/4ε²
  
  - Forward:
  
     On-diagonal terms: 𝛿²f/𝛿x<sub>j</sub>² = (f(x<sub>j,0</sub>+2ε)-2f(x<sub>j,0</sub>+ε)+f(x<sub>j,0</sub>))/ε²
     
     Off-diagonal terms: 𝛿²f/𝛿x<sub>j</sub>𝛿x<sub>k</sub> = (f(x<sub>j,0</sub>+ε,x<sub>k,0</sub>+ε)-f(x<sub>j,0</sub>+ε,x<sub>k,0</sub>)-f(x<sub>j,0</sub>,x<sub>k,0</sub>+ε)+f(x<sub>j,0</sub>,x<sub>k,0</sub>))/ε²
  
  - Backward:
  
     On-diagonal terms: 𝛿²f/𝛿x<sub>j</sub>² = (f(x<sub>j,0</sub>-2ε)-2f(x<sub>j,0</sub>-ε)+f(x<sub>j,0</sub>))/ε²
     
     Off-diagonal terms: 𝛿²f/𝛿x<sub>j</sub>𝛿x<sub>k</sub> = (f(x<sub>j,0</sub>-ε,x<sub>k,0</sub>-ε)-f(x<sub>j,0</sub>-ε,x<sub>k,0</sub>)-f(x<sub>j,0</sub>,x<sub>k,0</sub>-ε)+f(x<sub>j,0</sub>,x<sub>k,0</sub>))/ε²
    
Copyright © 2016 - Gabriel Sabença Gusmão

[![linkedin](https://static.licdn.com/scds/common/u/img/webpromo/btn_viewmy_160x25.png)](https://br.linkedin.com/pub/gabriel-saben%C3%A7a-gusm%C3%A3o/115/aa6/aa8)

[![researchgate](https://www.researchgate.net/images/public/profile_share_badge.png)](https://www.researchgate.net/profile/Gabriel_Gusmao?cp=shp)
