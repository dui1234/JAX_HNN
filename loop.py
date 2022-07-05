# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as np


def _while_loop_scan(cond_fun, body_fun):
    """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""

    def _iter(val):
        """Real iteration step : val->(next_val,next_cond)"""
        next_val = body_fun(val)
        next_cond = cond_fun(val, next_val)
        return next_val, next_cond

    def _fun(carry, it):
        """Scan function that will choose a real iteration or a dummy do-nothing function
        (value,cond)->it->((next_value,next_cond), next_it)
        """
        val, cond = carry
        # When cond is NOT met, we start doing no-ops.
        return jax.lax.cond(cond, _iter, lambda x: (x, False), val), cond

    def looper(init_val, max_iter):
        init = (init_val, True)
        ans, conds = jax.lax.scan(_fun, init, None, length=max_iter)
        return ans[0], np.logical_not(ans[1]), np.sum(conds)

    return looper


def _while_loop_python(cond_fun, body_fun):
    """Python based implementation (no jit, reverse-mode autodiff ok).
    Args:
        cond_fun : val -> next_val -> true|false
        body_fun : val -> next_val
    Returns:
        looper : val -> max_iter -> (val, bool, iter)
    """

    def looper(val, max_iter):
        for i in range(max_iter):
            newval = body_fun(val)
            cond = cond_fun(val, newval)
            if not cond:
                # When condition is not met, break (not jittable).
                break
            val = newval
        return val, np.logical_not(cond), i + 1

    return looper


def _while_loop_lax(cond_fun, body_fun):
    """lax.while_loop based implementation (jit by default, no reverse-mode)."""

    def _body_fun(_val):
        it, val, oldval = _val
        oldval = val
        val = body_fun(val)
        return it + 1, val, oldval

    def _cond_gen(max_iter):
        def _cond_fun(_val):
            it, val, oldval = _val
            return np.logical_or(np.logical_and(cond_fun(oldval, val), it <= max_iter - 1), it < 1)

        return _cond_fun

    def looper(val, max_iter):
        ans = jax.lax.while_loop(_cond_gen(max_iter), _body_fun, (0, val, val))
        return ans[1], np.logical_not(cond_fun(ans[2], ans[1])), ans[0]

    return looper


def while_loop(cond_fun, body_fun, *, unroll, jit):
    """Returns a while loop with a bounded number of iterations

    Will loop as long as cond_fun is met!

    Args:
        cond_fun : val -> next_val -> true|false
        body_fun : val -> next_val

        lax.scan       = unroll & jit
        lax.while_loop = jit
        python_loop    = unroll
    Returns:
        looper : val -> max_iter -> (val, bool, iter)

    """

    if unroll:
        if jit:
            fun = _while_loop_scan(cond_fun, body_fun)
        else:
            fun = _while_loop_python(cond_fun, body_fun)
    else:
        if jit:
            fun = _while_loop_lax(cond_fun, body_fun)
        else:
            raise ValueError("unroll=False and jit=False cannot be used together")

    if jit and fun is not _while_loop_lax:
        # jit of a lax while_loop is redundant, and this jit would only
        # constrain maxiter to be static where it is not required.
        fun = jax.jit(fun, static_argnums=1)

    return fun
