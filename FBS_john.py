from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import interp
import jaxopt
from functools import partial
from jax.experimental import ode
from jax import jit,grad,vmap
from loop import while_loop

def time_mirroring(t0,t1):
    """Time mirroring function for backward integration 
    Args:
        ts : tim grid
    Returns:
        function t->t to reverse/mirorr times 
    """
    return lambda t : t1 - t - t0 # time flip function

def optimal_control_setup(*,f,g,gbc,control,mixing,iter_crit,ts,x0,state_interpolator,control_interpolator):
    """
    Generate FBS solution to optimal control problem given state and value dynamics
        dx = f(t,x,u)
        dλ = g(t,x,u,λ)
    
    Args:
        f         : constraint dynamics t -> x -> u -> x'
        g         : value dynamics      t -> x -> u -> λ -> λ'
        gbc       : value boundary cond t -> x -> u -> λ
        control   : optimal control     t -> x -> λ -> u
        mixing    : mixing function     u -> u -> u
        iter_crit : iteration condition a -> a -> bool
        [ts]      : time grid over which solution should be generated (function interpolation points)
        x0        : initial condition for x
        state_interpolator   : x interpolator [t] -> ([x] -> t -> x)
        control_interpolator : u interpolator [t] -> ([u] -> t -> u)
        
    Returns:
        init  : u -> ([x],[λ],[u])
        cond  : ([x],[λ],[u]) -> ([x],[λ],[u]) -> bool
        body  : ([x],[λ],[u]) -> ([x],[λ],[u])
    """
    xinterp = state_interpolator(ts)         # x interpolating function
    uinterp = control_interpolator(ts)       # u interpolating function
    tflip   = time_mirroring(ts[0], ts[-1])  # time grid for reverse integration
    utrj    = vmap(control, in_axes=(0,0,0)) # mapped optimal control function
    
    # define ode rhs functions with odeint signature : x->t->*args->xdot 
    def dx(x,t,us): # forward integration
        """state dynamics θ->t->[k]->θ'
        """
        return f(t, x, uinterp(us, t)) 

    def dλ(λ,t,xs,us) : # backward integration
        """value dynamics λ->t->[x]->[u]->λ'
        """
        tprime = tflip(t)
        return -g(tprime, xinterp(xs, tprime), uinterp(us, tprime), λ)
    
    def body(val):
        """ Body function to perform one-sweep of the FBS algorithm
            ([x],[λ],[u]) -> ([x],[λ],[u])
        Args:
            val : tuple of values for x,λ,u
        Returns:
            val : updated tuple of values for x,λ,u
        """
        xs,λs,us = val
        # forward integration t:(0, Tmax)
        xs = ode.odeint(dx, x0, ts, us)
        # backward integration t:(Tmax, 0)
        λf= gbc(ts[-1],xs[-1], us[-1])
        λs= ode.odeint(dλ, λf, tflip(ts)[::-1], xs, us)[::-1]
        # update control
        us = mixing(us, utrj(ts, xs, λs))
        return xs, λs, us
    
    def init():
        """Initialization condition used to generate proper size/shape initial guesses
        """
        tshape = ts.shape
        xshape = tshape + np.array(x0).shape
        def func(k0):
            """Initialization function k->([θ],[λ],[k])"""
            k0 = np.array(k0)
            return np.zeros(xshape), np.zeros(xshape), np.ones(tshape+k0.shape)*k0  # x,λ,u
        return func

    def cond(val, newval):
        """ Continuation criteria for FBS 
            ([x],[λ],[u]) -> ([x],[λ],[u]) -> bool

        Checks convergence of x AND u
            Δiter(x,xnew) > δ and Δiter(u,unew) > δ

        Args:
            val   : old solution
            newval: new solution
        Returns:
            bool : True (iterations should continue) or False (iterations should stop = converged)
        """
        x,_,u       = val
        xnew,_,unew = newval
        return np.logical_and(iter_crit(x,xnew), iter_crit(u,unew))
    
    return init(), cond, body
  
def nashify(fun):
    """Nashified function, evaluated at θ=ψ, k=ϰ
    Args :
        fun : t -> θ -> ψ -> k -> ϰ -> *args -> x
    Returns:
        newfun : t -> θ -> k -> *args -> x
    """
    return lambda t,θ,k,*args : fun(t,θ,θ,k,k,*args)
def negative(fun):
    return lambda *args : -fun(*args) 
def square(fun):
    return lambda *args : np.linalg.norm(fun(*args))
def identity(fun):
    return lambda *args : fun(*args)
  
def mixfun(post, weight):
    """ Returns mixing function kold -> knew -> knew
        knew = (1-weight)*kold + weight*new
    """
    def fun(k,knew):
        k = post((1.0-weight)*k + weight*knew)
        return k
    return fun

def iteration_criteria(δ):
    def fun(val, newval):
        """ Iteration continuation criterium. 
        Checks if Δiter(val,newval) = |newval - val| / |newval| > δ
        Args:
            val   : old values
            newval: new values
        """
        return δ*np.linalg.norm(newval, ord=1) - np.linalg.norm(newval-val, ord=1) < 0
    return fun
  
def constrainedEL(state_index=1, control_index=2):
    """EL generatting function
        Args:
            state_index : coordinate position
            control_index: control variable position
    """
    def fun(V,W,F):
        """Dynamics for constrained EL optimal control problem
        Args:
            F : dynamic constraints   a1->a2->...->an->f
            V : cost/payoff function  a1->a2->...->an->v
            W : salvage function      a1->a2->...->an->w
            state_index   : argument position for constrained coordinates (1,...,n)
            control_index : argument position for control parameter (1,...,n)
        Returns:
            dλ   : lagrange multiplier dynamics a1->...->an->λ->λ'
            λbc  : boundary function for λ      a1->...->an->λ
            kopt : optimality condition         a1->...->an->λ->c
                   kopt = 0 at the optimal control
        """
        def lagrangian(*args):
            """Augmented Lagrangian/Hamiltonian L'(...,λ) = V(...) + λ.F(...) """
            return V(*args[:-1]) + np.dot(args[-1], F(*args[:-1]))

        return F, negative(grad(lagrangian, argnums=state_index)), negative(grad(W, argnums=state_index)), grad(lagrangian, argnums=control_index)
    return fun

def nashEL(V,W,F):
    """Given cost/payoff function,  for the population/individual optimal control problem, return EL equations under Nash equiblirum
    Args:
        V cost       : t -> θ -> ψ -> k -> ϰ -> ... -> v
        W salvage    : t -> θ -> ψ -> k -> ϰ -> ... -> w
        F constraint : t -> θ -> ψ -> k -> ϰ -> ... -> f
    Returns:
        dθ   : t -> θ -> k -> ... -> θ'
        dλ   : t -> θ -> k -> ... -> λ -> λ'
        λbc  : t -> θ -> k -> ... -> λ        
        kopt : t -> θ -> k -> ... -> λ -> c
    """
    _, dλ,λbc,kopt = constrainedEL(2,4)(V,W,F)
    return nashify(F), nashify(dλ), nashify(λbc), nashify(kopt)

def optimal_control(optimality,optimizer,**kwargs):
    """Return optimal control function generator given optimality condition
    Args:
        optimality : t->x->u->...-> c (=0 for optimal control)
        optimizer  : jaxopt optimizer
        kwargs     : kwargs for jaxopt optimizer
    Returns:
        uopt u0 -> (t->x->...->λ->u)
    """
    solver = optimizer(fun = lambda u,t,x,*args : optimality(t,x,u,*args), **kwargs)
    def solver_guess(guess):
        u0 = np.array(guess)
        def fun(t,x,*args):
            opt, _ = solver.run(u0, t, x, *args)
            return opt
        return fun
    return solver_guess

def solver(*, f, g, gbc, control, times, x0, uguess, mixing, state_interpolator, control_interpolator, unroll, jit, maxiter=1000,delta=1e-6):
    """Solve optimal control problem given state and value dynamics and boundary conditons
    Args:
        f                   : state dynamics
        g                   : value dynamics
        gbc                 : value end-boundary conditions
        times               : [ts] values for solution
        x0                  : initial state
        uguess              : control guess
        mixing              : mixing function for control
        state_interpolator  : interpolation function for states
        control_interpolator: interpolation function for control
    """
    init, cond, body = optimal_control_setup(f=f, g=g, gbc=gbc, control=control, \
                                             mixing=mixing, iter_crit=iteration_criteria(delta), \
                                             ts=times, x0 = np.array(x0),\
                                             state_interpolator=state_interpolator,\
                                             control_interpolator=control_interpolator)
    solver = while_loop(cond, body, unroll=unroll, jit=jit)
    (x,λ,u), conv, nit = solver(init(uguess), maxiter)
    print(f"Converged = {conv}, Iterations = {nit:4d}")
    return (x,λ,u), (init, cond, body, solver)

def generic_solver(*, LagrangeEquations, V, W, F, **kwargs):
    f, dλ, λbc, optimality = LagrangeEquations(V, W, F)
    opt = optimal_control(square(optimality), jaxopt.GradientDescent, tol=1e-11)
    return solver(f=f, g=dλ, gbc=λbc, control=opt(kwargs['uguess']), **kwargs)

def maxerror(a,b):
    return np.max(np.abs(a-b))
