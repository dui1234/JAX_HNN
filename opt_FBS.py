import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from jax import lax
import interp
import jaxopt
from jax import jit

def negative(fun):
    return lambda *args : -fun(*args)

def time_back(t0,t1):
  return lambda t : t1 - t - t0

def square(fun):
    return lambda *args : jnp.linalg.norm(fun(*args))

def maxerror(a,b):
    return np.max(np.abs(a-b))
	
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
def constrainedEL(V,W,F):
  Aug_lagrange = lambda t,q,u,lamb: V(t,q,u) + jnp.dot(lamb,F(t,q,u))
  return F, negative(jax.grad(Aug_lagrange,1)), jax.grad(W,1), jax.grad(Aug_lagrange,2)

def optimal_control(optimality, optimizer):
  opti_func = lambda u,t,q,lamb: optimality(t,q,u,lamb)
  solver = optimizer(opti_func, tol=1e-11)
  def u_opt(u0):
    def solv(t,q,lamb):
        u_opt_sol, _ = solver.run(u0,t,q,lamb)
        return u_opt_sol
    return solv
  return u_opt

check = jit(lambda x_old,x_new,delta: (delta)*jnp.linalg.norm(x_new,ord=1) - jnp.linalg.norm(x_new - x_old,ord=1))

def iteration_criteria(δ):
  def fun(val, newval):
    return δ*np.linalg.norm(newval, ord=1) - np.linalg.norm(newval-val, ord=1) < 0
  return fun
#/////////////////////////////////////////////////////////////////////////////////////////////////////////

def body_function_generator(V,W,F,times,u0,x0,q_interp,u_interp):
  f,dlambda,lambda_bc,uopt = constrainedEL(V,W,F)
  opt = optimal_control(square(uopt),jaxopt.GradientDescent)
  control = jit(jax.vmap(opt(u0),in_axes=(0,0,0))) # <--(t,q,λ) 

  tshape = times.shape
  xshape = tshape + jnp.array(x0).shape

  xs = jnp.zeros(xshape)
  λs = jnp.zeros(xshape)
  us = jnp.ones(tshape+u0.shape)*u0

  t_back   = time_back(times[0], times[-1])

  q_interporator = q_interp(times)
  u_interporator = u_interp(times)

  dx = jit(lambda x,t,us: F(t,x,u_interporator(us,t)))
  dλ = jit(lambda λ,t,xs,us: -dlambda(t_back(t),q_interporator(xs,t_back(t)),u_interporator(us,t_back(t)),λ))

  return dx,dλ,xs,λs,us,control,opt,lambda_bc

def loop_sol(V,W,F,times,u0,x0,mixing,q_interp,u_interp,delta):
  dx,dλ,xs,λs,us,control,opt,lambda_bc = body_function_generator(V,W,F,times,u0,x0,q_interp,u_interp)
  cond = iteration_criteria(1e-6)

  count = 0
  condition2 = -100.0
  #condition3 = True
  
  while(condition2 < 0):
    u_old = us
    x_old = xs
    λ_old = λs

    xs = odeint(dx,x0,times,us)
    λf = lambda_bc(times[-1],xs[-1],us[-1])
    λs = odeint(dλ,λf,times,xs,us)[::-1]

    u_t = control(times,xs,λs)
    #us = 0.1*u_t + (1.0 - 0.1)*us
    us = mixing(us, u_t)

    check1 = check(u_old,us,delta)
    check2 = check(x_old,xs,delta)
    check3 = check(λ_old,λs,delta)
    #condition = jnp.logical_and(check1<0,jnp.logical_and(check2<0,check3<0))
    condition2 = jnp.min(jnp.array([check1,check2,check3]))
    #condition3 = jnp.logical_and(cond(x_old,xs), cond(u_old,us))
    #print(count,check1,check1>1e-6,check2,check2>1e-6,check3,check3>1e-6,condition,condition2,condition2<0,condition3)
    count +=1
    if(count > 200):
      if count%100 == 0: print(f'loop over 200 iterations: {count}')
    if count > 1001:
      print(f'loop over 1000 iterations')
      return (xs,us,λs) , count
  return (xs,us,λs) , count

def mixfun(post, weight):
  def fun(k,knew):
    k = post((1.0-weight)*k + weight*knew)
    return k
  return fun
