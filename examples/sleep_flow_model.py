
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    
    def __init__(self, key):
        # 5D input (Delta, Theta, Alpha, Sigma, Beta) -> 5D output (dx/dt)
        self.mlp = eqx.nn.MLP(
            in_size=5,
            out_size=5,
            width_size=64,
            depth=3,
            activation=jax.nn.softplus, # Softplus for smooth derivatives
            key=key
        )

    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    vector_field: VectorField

    def __init__(self, key):
        self.vector_field = VectorField(key)

    def __call__(self, y0, t_eval):
        # Solves the ODE starting from y0 at times t_eval
        # t_eval[0] should be the start time
        
        term = diffrax.ODETerm(self.vector_field)
        solver = diffrax.Tsit5() # Runge-Kutta 5(4)
        # Use PIDController for adaptive step size
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        
        # We want to solve from t_eval[0] to t_eval[-1]
        t0 = t_eval[0]
        t1 = t_eval[-1]
        dt0 = 0.1 # Initial step guess
        
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(ts=t_eval)
        )
        
        return sol.ys
