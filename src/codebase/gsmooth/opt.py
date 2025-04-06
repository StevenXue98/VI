import jax
import jax.numpy as jnp
from functools import partial
import time
from tqdm import tqdm  # Preferred progress bar for JAX

from .smoothers import IdentityFilter


class FilteredOptimization(object):
    def __init__(self, grad_fun, init_params,
                 grad_filter=IdentityFilter(),
                 step_size=0.001,
                 callback=None,
                 fun=None,
                 true_grad_fun=None,
                 follow_true_grad=False,
                 save_params=False,
                 save_grads=False,
                 save_filtered_grads=False,
                 num_marginal_samples_to_save=0):
        """ Optimization object that uses a "gradient filter" --- this filter
        keeps a running estimate of the "true" gradient, and has
        filter-specific update rules (see AdamFilter, SGDMomentumFilter)

        Usage

          opt_obj = FilteredOptimization(grad_lnpdf, init_params,
                                         grad_filter=AdamFilter(),
                                         step_size=0.001,
                                         callback=mycallback)
          opt_obj.run(num_iters=100)

        Args:

            num_marginal_samples_to_save : int, num samples to save at each
                iteration, or tuple (num_samps, num_skip) --- save marginal
                gradient samples only every num_skip iterations
        """
        self.grad_fun = grad_fun
        self.grad_filter = grad_filter
        self.init_params = jnp.array(init_params)  # Ensure JAX array
        self.params = jnp.array(init_params)  # Ensure JAX array

        if callback is None:
            self.callback = self.default_callback
        else:
            self.callback = callback

        self.t = 0
        self.step_size = step_size

        # save params
        self.save_params = save_params
        self.param_trace = []

        # machinery for saving gradient + function value traces
        self.save_grads = save_grads
        self.grad_trace = []

        # filtered grads
        self.save_filtered_grads = save_filtered_grads
        self.filtered_grad_trace = []

        # true gradients (requires function)
        self.true_grad_fun = true_grad_fun
        self.true_grad_trace = []

        # function value
        self.fun = fun
        self.fun_vals = []

        # samples of the gradient at each marginal
        if isinstance(num_marginal_samples_to_save, tuple):
            nms, nskip = num_marginal_samples_to_save
        else:
            nms, nskip = num_marginal_samples_to_save, 1
        self.num_marginal_samples_to_save = nms
        self.marginal_sample_skip = nskip
        self.marginal_samples = {}

        # timer
        self.wall_clock = 0.0

    def run(self, num_iters, step_size=None, key=None):
        """Run optimization for a specified number of iterations"""
        # Update step size if appropriate
        if step_size is not None:
            self.step_size = step_size
        ss = self.step_size

        # Initialize random key if not provided
        if key is None:
            key = jax.random.PRNGKey(0)

        start_time = time.time()

        for i in range(num_iters):
            # Get noisy gradient
            noisy_grad = self.grad_fun(self.params, self.t)

            # Update parameters using the gradient filter
            self.params, filtered_grad = self.grad_filter.update(
                self.params,
                noisy_gradient=noisy_grad,
                step_size=ss
            )

            # Save and track progress
            self.track_progress(noisy_grad, filtered_grad)

            # Increment optimization step count
            self.t += 1

        # Track wall clock time
        self.wall_clock += (time.time() - start_time)

    def track_progress(self, noisy_grad, filtered_grad):
        """Track and save optimization progress"""
        # If function provided, save values
        if self.fun is not None:
            self.fun_vals.append(float(self.fun(self.params, self.t)))

        # Report on gradient via callback
        if self.callback is not None:
            self.callback(self.params, self.t, noisy_grad)

        # Update object attributes based on what needs to be saved
        if self.save_params:
            # Convert to numpy for storage to avoid JAX device arrays in lists
            self.param_trace.append(jnp.array(self.params))

        if self.save_grads:
            self.grad_trace.append(jnp.array(noisy_grad))

        if self.save_filtered_grads:
            self.filtered_grad_trace.append(jnp.array(filtered_grad))

        if self.true_grad_fun is not None:
            true_grad = self.true_grad_fun(self.params, self.t)
            self.true_grad_trace.append(jnp.array(true_grad))

        if (self.num_marginal_samples_to_save > 0) and \
                (self.t % self.marginal_sample_skip == 0):
            nms = self.num_marginal_samples_to_save
            print(f"  ... saving {nms} marginal samples (iter {self.t})")

            # Collect samples
            msamps = []
            for _ in tqdm(range(nms)):
                msamps.append(self.grad_fun(self.params, self.t))

            self.marginal_samples[self.t] = jnp.array(msamps)

    def default_callback(self, th, t, g):
        """Default callback to print progress"""
        if t % 20 == 0:
            if self.fun is not None:
                fval = self.fun(th, t)
                print(f"iter {t}: val = {fval:2.4f}, gmag = {jnp.sqrt(jnp.dot(g, g)):2.4f}")
            else:
                print(f"iter {t}: gmag = {jnp.sqrt(jnp.dot(g, g)):2.4f}")

    def plot_gradients(self, dims=[1, 10, 50]):
        """Plot gradient traces for visualization"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("white")
        except ImportError:
            print("Matplotlib and/or seaborn not available for plotting")
            return None, None

        noisy_grads = jnp.array(self.grad_trace)
        true_grads = jnp.array(self.true_grad_trace)
        filt_grads = jnp.array([g[0] for g in self.filtered_grad_trace])
        filt_grads_var = jnp.array([g[1] for g in self.filtered_grad_trace])
        tgrid = jnp.arange(true_grads.shape[0])

        fig, axarr = plt.subplots(len(dims), 1, figsize=(12, 3 * len(dims)))
        axarr = axarr if len(dims) > 1 else [axarr]  # Handle single dimension case

        for d, ax in zip(dims, axarr):
            ax.plot(tgrid, true_grads[:, d], label="true")
            ax.plot(tgrid, filt_grads[:, d], label="filtered")
            ax.fill_between(
                tgrid,
                filt_grads[:, d] - 2 * jnp.sqrt(filt_grads_var[:, d]),
                filt_grads[:, d] + 2 * jnp.sqrt(filt_grads_var[:, d]),
                alpha=0.25
            )
            ax.scatter(tgrid, noisy_grads[:, d], s=3)
            ax.set_xlim((tgrid[0], tgrid[-1]))

        axarr[-1].legend()

        print("Adam average grad deviation:",
              jnp.sqrt(jnp.mean((filt_grads - true_grads) ** 2)))
        print("sample average deviation:",
              jnp.sqrt(jnp.mean((noisy_grads - true_grads) ** 2)))

        return fig, axarr


class ControlVariateOptimization(FilteredOptimization):
    def __init__(self, grad_fun, cv_grad_fun, init_params, D,
                 mc_and_cv_gfun=None, **kwargs):
        """ Optimization object that uses a "gradient filter" --- this filter
        keeps a running estimate of the "true" gradient, and has
        filter-specific update rules (see AdamFilter, SGDMomentumFilter)

        Args:
            grad_fun: function that takes in randomness
                        noisy_grad = grad_fun(lam, eps)
            cv_grad_fun: function that returns
                        noisy_grad_approx, gmean, gvar = cv_grad_fun(lam, eps)
            init_params: lambda_0
        """
        super(ControlVariateOptimization, self).__init__(grad_fun, init_params, **kwargs)

        ## set up control variate parameters
        self.D = D
        self.cv_grad_fun = cv_grad_fun
        self.mc_and_cv_gfun = mc_and_cv_gfun
        self.beta_rho, self.beta_eta = 0.999, 0.99
        self.rho, self.eta = 0.0, 0.0

        ## record cv marginal variances
        self.cv_grad_samps = {}
        self.mc_grad_samps = {}
        self.cv_var = {}
        self.mc_var = {}
        self.frac_var = {}
        self.elementwise_corrs = {}
        self.elementwise_covs = {}
        self.c_stars = {}
        self.ideal_c_star = {}
        self.estimated_covs = {}
        self.ideal_frac_var = {}
        self.ideal_frac_magnitude_var = {}
        self.frac_magnitude_var = {}

    def compute_c_star(self, m_tilde, v_tilde, covs=None, cthresh=1.0):
        """Compute optimal control variate coefficient"""
        if covs is not None:
            return covs / v_tilde

        # otherwise running estimate
        c_star = (self.rho - m_tilde * self.eta) / v_tilde
        c_star = jnp.clip(c_star, -cthresh, cthresh)
        return c_star

    def step(self, L, step_size=None, cv_scale=1.0, track_statistics=False, key=None):
        """ Run a single step """
        ss = self.step_size if step_size is None else step_size

        # Initialize random key if not provided
        if key is None:
            key = jax.random.PRNGKey(0)

        # Generate noise
        eps = jax.random.normal(key, (L, self.D))

        if self.mc_and_cv_gfun is None:
            gsamp, gsamp_var = self.grad_fun(self.params, eps)
            gsamp_tilde, m_tilde, v_tilde = self.cv_grad_fun(self.params, eps)
            covs = None
        else:
            gsamp, gsamp_tilde, m_tilde, v_tilde, covs, dvar = \
                self.mc_and_cv_gfun(self.params, eps)

        # compute optimal scaling
        c_star = cv_scale * self.compute_c_star(m_tilde, v_tilde, covs)
        gsamp_cv = gsamp - c_star * (gsamp_tilde - m_tilde)

        # track noisy estimate of E[noisy_g] and E[noisy_g * g_tilde]
        self.eta = self.beta_eta * self.eta + (1.0 - self.beta_eta) * gsamp
        self.rho = self.beta_rho * self.rho + (1.0 - self.beta_rho) * (gsamp * gsamp_tilde)

        # now plug in gsamp_cv into the update filter
        self.params, filtered_grad = \
            self.grad_filter.update(self.params,
                                    noisy_gradient=gsamp_cv,
                                    step_size=ss)

        # save params
        self.track_progress(gsamp_cv, filtered_grad)

        if track_statistics:
            stats_key = jax.random.fold_in(key, self.t)  # Derive new key
            self.track_marginal_variances(L, L - 2, c_star=c_star,
                                          covs=covs, num_samps=200, key=stats_key)

        # increment num opt steps
        self.t += 1

        # Return new key for state-based usage
        return jax.random.fold_in(key, 1)

    def run(self, num_iters, L, step_size=None, key=None):
        """ Run num_iters steps of the optimizer """
        # first update step size if appropriate
        if step_size is not None:
            self.step_size = step_size
        ss = self.step_size

        # Initialize random key if not provided
        if key is None:
            key = jax.random.PRNGKey(0)

        for i in range(num_iters):
            key = jax.random.fold_in(key, i)
            key = self.step(L, step_size=ss, key=key)

    def track_marginal_variances(self, L, L_cv, c_star, covs, num_samps=1000, key=None):
        """Track and record marginal variances for evaluation"""
        assert L_cv < L, "number of samples used in CV must be smaller for fair comparison"

        # Initialize random key if not provided
        if key is None:
            key = jax.random.PRNGKey(0)

        # epsilon mat sample
        eps_mat = jax.random.normal(key, (num_samps, L, self.D))

        # Collect gradient samples
        gsamps = []
        for eps in eps_mat:
            gsamps.append(self.grad_fun(self.params, eps)[0])
        gsamps = jnp.array(gsamps)

        # Get control variate samples
        gsamps_tilde, m_tilde, v_tilde = \
            self.cv_grad_fun(self.params, eps_mat[:, :L_cv, :])

        # Compute correlations between gsamps and gsamps_tilde
        self.elementwise_corrs[self.t] = jnp.array([
            jnp.corrcoef(gsamps_tilde[:, idx], gsamps[:, idx])[0, 1]
            for idx in range(gsamps.shape[1])
        ])

        self.elementwise_covs[self.t] = jnp.array([
            jnp.cov(gsamps_tilde[:, idx], gsamps[:, idx])[0, 1]
            for idx in range(gsamps.shape[1])
        ])

        self.estimated_covs[self.t] = covs

        # Compute optimization cv
        gsamps_cv = gsamps - c_star * (gsamps_tilde - m_tilde)

        self.cv_grad_samps[self.t] = gsamps_cv
        self.mc_grad_samps[self.t] = gsamps
        self.c_stars[self.t] = c_star

        self.cv_var[self.t] = jnp.var(gsamps_cv, axis=0)
        self.mc_var[self.t] = jnp.var(gsamps, axis=0)
        self.frac_var[self.t] = jnp.var(gsamps_cv, axis=0) / jnp.var(gsamps, axis=0)
        self.ideal_c_star[self.t] = self.elementwise_covs[self.t] / v_tilde

        # Best possible?
        gsamps_cv_ideal = gsamps - self.ideal_c_star[self.t] * (gsamps_tilde - m_tilde)
        self.ideal_frac_var[self.t] = jnp.var(gsamps_cv_ideal, axis=0) / jnp.var(gsamps, axis=0)

        # Gradient magnitudes
        def mags(x):
            return jnp.sqrt(jnp.sum(x ** 2, axis=1))

        self.frac_magnitude_var[self.t] = \
            jnp.var(mags(gsamps_cv)) / jnp.var(mags(gsamps))

        self.ideal_frac_magnitude_var[self.t] = \
            jnp.var(mags(gsamps_cv_ideal)) / jnp.var(mags(gsamps))

    def default_callback(self, th, t, g):
        """Default callback to print progress"""
        if t % 20 == 0:
            if self.fun is not None:
                fval = self.fun(th, t)
                print(f"iter {t}: val = {fval:2.4f}, gmag = {jnp.sqrt(jnp.dot(g, g)):2.4f}")
            else:
                print(f"iter {t}: gmag = {jnp.sqrt(jnp.dot(g, g)):2.4f}")