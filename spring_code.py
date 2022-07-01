from jax.config import config; config.update("jax_enable_x64",True)
import jax 
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from jax.experimental.ode import odeint

def hamiltonian_fxn(q,p):
  return p*p + q*q

def dH(coors,t):
  q,p = coors
  dHdq,dHdp = grad(hamiltonian_fxn,argnums=(0,1))(q,p)
  S = jnp.array([dHdp, -dHdq])
  return S

def get_trajectory(t_span = [0,3], timescale = 10, radius = 100, y0 = np.array([100,100]), noise_std = 0.1, **kwargs):
  t_eval = jnp.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0]))) #The time steps

  #get initial state
  #if y0 is None (i.e, 100): y0 = jnp.array(np.random.rand(2)*2 - 1)
  #if radius is None (i.e, 100): raduis = jnp.array(np.random.rand()*0.9 + 0.1)
  y0_dmy = jnp.array(np.random.rand(2)*2 - 1)
  radius_dmy = jnp.array(np.random.rand()*0.9 + 0.1)
  y0 = jnp.where(y0[0] == 100, y0_dmy, y0)
  radius = jnp.where(radius ==  100, radius_dmy, radius)

  y0 = y0/ jnp.sqrt((y0**2).sum()) * radius
  
  spring_ivp = odeint(dH,y0,t_eval, rtol=1e-10)
  q, p = spring_ivp[:,0], spring_ivp[:,1]
  #dydt = [dH(y, None) for y in spring_ivp] #replaced by vmap
  #dydt = jnp.stack(dydt)
  dydt = vmap(dH,in_axes=(0,0))(spring_ivp,None)
  dqdt, dpdt = dydt[:,0],dydt[:,1]

  # add noise
  q += np.random.randn(*q.shape)*noise_std
  p += np.random.randn(*p.shape)*noise_std

  return q, p, dqdt, dpdt, t_eval

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
  field = {'meta': locals()}

  # meshgrid to get vector field
  b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
  ys = jnp.stack([b.flatten(), a.flatten()])

  # get vector directions
  #dydt = [dH(y, None) for y in ys.T]
  #dydt = jnp.stack(dydt)  #replaced by vmap
  dydt = vmap(dH,in_axes=(0,0))(ys.T,None)

  field['x'] = ys.T
  field['dx'] = dydt
  return field

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
  data = {'meta': locals()}

  # randomly sample inputs
  np.random.seed(seed)
  xs, dxs = [], []

  for s in range(samples):
    x, y, dx, dy, t  = get_trajectory(**kwargs)
    xs.append( np.stack( [x, y]).T )
    dxs.append( np.stack( [dx, dy]).T )

  data['x'] = jnp.concatenate(xs)
  data['dx'] = jnp.concatenate(dxs).squeeze()

  # make a train/test split
  split_ix = int(len(data['x']) * test_split)
  split_data = {}
  for k in ['x', 'dx']:
    split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
  data = split_data
  return data
