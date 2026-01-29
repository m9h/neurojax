import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import seaborn as sns
from neurojax.models.physics import WongWang
from neurojax.dynamics import SINDyOptimizer, polynomial_library

def demo_sindy_nmm():
    print("Initializing Wong-Wang Neural Mass Model...")
    # 1. Setup Model and Simulate Data
    # Wong-Wang parameters
    # The default WongWang class requires 'G' param.
    model = WongWang(G=0.5) 
    
    # We need to wrap the model's vector field for diffrax
    def vector_field(t, y, args):
        return model.vector_field(t, y, args)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    t0, t1 = 0.0, 2000.0 # ms
    dt = 1.0 # 1ms steps
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    
    # Initial state (Synaptic gating S)
    y0 = jnp.array([0.1]) 
    
    # Args: (I_net, I_ext)
    # Let's provide some constant input to verify fixed point or limit cycle
    # I_net = 0, I_ext = 0.3 (noise-free for SINDy first)
    args = (0.0, 0.3)
    
    print("Simulating (T=2000ms)...")
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt, y0, args=args, saveat=saveat)
    X = sol.ys # Shape (2000, 1) - single variable S
    
    # Compute derivative dS/dt from model physics directly or finite difference
    # Let's use exact model derivative to give SINDy the best chance
    dX = jax.vmap(lambda s: vector_field(0, s, args))(X)
    
    # 2. Run SINDy
    print("Running SINDy...")
    # Polynomial library up to degree 5 to capture the sigmoid nonlinearity Taylor expansion?
    # The Wong-Wang has a sigmoid H(x). 
    # H(x) = (a - b*x) / (1 + exp(-d*(e*x - f)))
    # This is HIGHLY nonlinear and not polynomial.
    # Standard SINDy with polynomial library will likely fit a Taylor approximation.
    
    optimizer = SINDyOptimizer(threshold=0.0001, max_iter=10)
    # We use a higher degree library
    fit_library = lambda x: polynomial_library(x, degree=5)
    
    Xi = optimizer.fit(X, dX, fit_library)
    
    print("\n--- SINDy Results ---")
    print("Coefficient Matrix Xi (Shape: {}, {})".format(*Xi.shape))
    print(Xi)
    
    # Interpret coefficients
    # Features: 1, S, S^2, S^3, S^4, S^5
    feature_names = ["1", "S", "S^2", "S^3", "S^4", "S^5"]
    print("\nDiscovered Equation:")
    eqn_str = "dS/dt = "
    for i, coeff in enumerate(Xi[:, 0]):
        if abs(coeff) > 1e-5:
            eqn_str += f"{coeff:.5f} * {feature_names[i]} + "
    print(eqn_str.strip(" + "))
    
    # 3. Validation / Reconstruction
    print("\nReconstructing dynamics from learned coefficients...")
    dX_pred = optimizer.predict(X, Xi, fit_library)
    
    mse = jnp.mean((dX - dX_pred)**2)
    print(f"MSE Prediction Error: {mse:.6f}")
    
    # Plotting (if environment allows, usually we just save figure)
    # plt.figure(figsize=(10, 5))
    # plt.plot(sol.ts, dX, label='True dS/dt')
    # plt.plot(sol.ts, dX_pred, '--', label='SINDy dS/dt')
    # plt.legend()
    # plt.savefig('sindy_nmm_fit.png')
    
    return Xi, mse

if __name__ == "__main__":
    demo_sindy_nmm()
