"""
Simple example script fitting a model for synthetic data
"""
from __future__ import print_function
import jax.numpy as jnp
from jax import jit, random
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these are available in JAX or custom implemented
# You'll need to reimplement these classes based on your specific requirements
from src.codebase.vi.BBVI import DiagMvnBBVI
from src.codebase.gsmooth.opt import FilteredOptimization
from src.codebase.gsmooth.smoothers import AdamFilter
import CV as cvg
import models

#########################################
# construct model function + vb object  #
#########################################
# lnpdf, D = make_model()
lnpdf, D, name = models.set_lnpdf("frisk")
key = random.PRNGKey(0)
th0 = random.normal(key, (D,))
print(lnpdf(th0))  # example use

# create bbvi object --- this just keeps references to lnpdf,
# grad(lnpdf), hvp(lnpdf), etc
vbobj = DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=False)

# initialize params
key = random.PRNGKey(1)
lam0 = random.normal(key, (vbobj.num_variational_params,)) * 0.01 - 1
lam0 = lam0.at[D:].set(-3.)

################################
# set up optimization function #
################################
n_samps = 2
n_iters = 800
step_size = 0.05


def run_timed_opt(gradfun, num_iters):
    """ runs num_iters without computing intermediate values,
    then computes 2000 sample elbo values (for timing)
    """
    mc_opt = FilteredOptimization(
        grad_fun=gradfun,
        init_params=lam0.copy(),
        save_params=True,
        save_grads=False,
        grad_filter=AdamFilter(),
        fun=lambda lam, t: 0.,
        callback=lambda th, t, g: 0.)

    print("  ... optimizing ")
    start_time = time.time()
    mc_opt.run(num_iters=num_iters, step_size=step_size)
    wall_clock = time.time() - start_time
    mc_opt.wall_clock = wall_clock
    print(f"  ... wall time: {wall_clock:2.4f}")

    print("computing ELBO values")
    # compute ~ 50 equally spaced elbo values here
    skip = 16
    fun_vals = jnp.array([vbobj.elbo_mc(lam, n_samps=500)
                          for lam in mc_opt.param_trace[::skip]])

    return fun_vals, mc_opt.wall_clock, mc_opt


#################################################
# define pure MC gradient function and optimize #
#################################################
print(f"\n ======== running MC, nsamps = {n_samps} =======")


@jit
def mc_grad_fun_inner(lam, eps):
    return -1. * cvg.construct_cv_grads(vbobj, lam, eps, method="mc").mean(0)


def mc_grad_fun(lam, t):
    key = random.PRNGKey(int(t))
    eps = random.normal(key, (n_samps, D))
    return mc_grad_fun_inner(lam, eps)


mc_vals, mc_wall_time, mc_opt = run_timed_opt(mc_grad_fun, num_iters=3 * n_iters)  # about 3 x for non hvp

################################################
# define RV-RGE gradient function and optimize #
################################################
print(f"\n ======= running CV, nsamps = {n_samps} ======")


@jit
def cv_gfun_inner(lam, eps):
    return -1. * cvg.construct_cv_grads(vbobj, lam, eps, method="hvp_with_loo_diag_approx").mean(0)


def cv_gfun(lam, t):
    key = random.PRNGKey(int(t))
    eps = random.normal(key, (n_samps, D))
    return cv_gfun_inner(lam, eps)


cv_vals, cv_wall_time, cv_opt = run_timed_opt(cv_gfun, num_iters=n_iters)

################
# plot results #
################
sns.set_style("white")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mc_vals, label="MC")
ax.plot(cv_vals, label="RV-RGE")
ax.set_ylim(mc_vals[-1] - 20, cv_vals[-1] + 10)
ax.legend(loc='best')
ax.set_xlabel("iteration")
ax.set_ylabel("ELBO")
ax.set_title(f"MC vs RV-RGE Comparison, step size = {step_size:2.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(jnp.linspace(0, mc_wall_time, len(mc_vals)), mc_vals, label="MC")
ax.plot(jnp.linspace(0, cv_wall_time, len(cv_vals)), cv_vals, label="RV-RGE")
ax.set_ylim(mc_vals[-1] - 20, cv_vals[-1] + 10)
ax.legend(loc='best')
ax.set_xlabel("wall clock (seconds)")
ax.set_ylabel("ELBO")
ax.set_title(f"MC vs RV-RGE Comparison, step size = {step_size:2.3f}")