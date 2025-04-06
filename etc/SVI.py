from etc.VI import *

class SVI(VI):
    '''
    Using naive doubly stochastic gradient
    '''
    def run(self, step_size=1e-3, seed=1, opt='adam', batch_size=5, num_iters=10000,
                    init_sigma=0.001, local_reparam=False, log_frequency=100):
        key = PRNGKey(seed)
        key, _key = split(key)
        loc, log_scale = (
            jax.random.normal(_key, self.flattened_param_template.shape) / 100,
            jnp.ones_like(self.flattened_param_template) * init_sigma,
        )
        params = {"loc": loc, "log_scale": log_scale}
        losses = []
        grad_norms = []
        optimizer = get_optimizer(opt, step_size)
        opt_state = optimizer.init(params)
        iter_counter = 0
        while iter_counter <= num_iters:
            key, _key = split(key)
            shuffled_idx = generate_batch_index(_key, self.dataset['N'], batch_size)
            for idx in shuffled_idx:
                key, _key = split(key)
                loss, grads, eps = self.get_loss_eps_grad(_key, params, idx, local_reparam)
                grad_norms.append(
                    (grads['loc'] ** 2).mean()
                )
                grads = tree_map(lambda g: g.mean(0), grads)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                if iter_counter % log_frequency == 0:
                    key, _key = split(key)
                    losses.append(self.eval_fulldataset_loss(_key, params))
                iter_counter += 1
        return params, losses, np.array(grad_norms)