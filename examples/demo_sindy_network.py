import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from neurojax.models.physics import WongWang
from neurojax.dynamics import SINDyOptimizer, polynomial_library

def demo_sindy_network():
    print("Initializing 2-Node Coupled Simulation...")
    # Parameters
    G = 1.0 # Global coupling scaling
    C = jnp.array([[0.0, 0.5],  # Node 1 receives 0.5 from Node 2
                   [0.2, 0.0]]) # Node 2 receives 0.2 from Node 1
    
    # Define Network Vector Field
    # y shape: (2,) for S1, S2
    # I_ext shape: (2,)
    model = WongWang(G=G)
    
    def network_field(t, y, args):
        I_ext = args
        # Calculate Input I_net = G * Sum(C_ij * S_j)
        # y is S vector.
        # I_net_i = G * Sum_j (C_ij * S_j)
        # Matrix multiply: I_net = G * (C @ y)
        I_net = G * (C @ y)
        
        # We need to map model.vector_field which takes scalar/vector args.
        # WongWang.vector_field expects (I_net, I_ext).
        # We can vmap the physics over the 2 nodes.
        
        # dS = -S/tau + ...
        # Standard WongWang implementation might be scalar. Let's check provided usage.
        # In current physics.py, WongWang is scalar-ish but JAX handles vectorization automatically if parameters allow.
        
        # Let's write the explicit coupled equations for clarity in this demo wrapper
        S = y
        x = model.w * I_net + I_ext
        H = (310 - 125*x) / (1 + jnp.exp(-2.6 * (125*x - 55)))
        dS = -S/model.tau_s + (1 - S) * model.gamma * H
        return dS

    # Simulation
    t0, t1 = 0.0, 2000.0
    dt = 1.0
    term = diffrax.ODETerm(network_field)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    
    y0 = jnp.array([0.1, 0.1])
    I_ext = jnp.array([0.3, 0.3]) # Excitatory drive
    
    print("Simulating Coupled System...")
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt, y0, args=I_ext, saveat=saveat)
    X = sol.ys # (T, 2)
    dX = jax.vmap(lambda s: network_field(0, s, I_ext))(X) # (T, 2)
    
    # SINDy Recovery
    print("Running SINDy on Network Data...")
    optimizer = SINDyOptimizer(threshold=0.001) # Lower threshold for coupling terms
    fit_library = lambda x: polynomial_library(x, degree=2) # Linear + Quadratic interactions
    
    Xi = optimizer.fit(X, dX, fit_library)
    
    print("\n--- Network Recovery Results ---")
    # Library: 1, S1, S2, S1^2, S1S2, S2^2
    names = ["1", "S1", "S2", "S1^2", "S1*S2", "S2^2"]
    
    for node in range(2):
        print(f"\nNode {node+1} (dS_{node+1}/dt):")
        eqn = ""
        for i, val in enumerate(Xi[:, node]):
            if abs(val) > 1e-4:
                eqn += f"{val:.5f}*{names[i]} + "
        print(eqn.strip(" + "))
        
    print("\nanalysis: Coupling Terms")
    # For Node 1, coupling comes from S2. Look for linear S2 term or S2 interactions.
    # Note: WongWang coupling enters inside the sigmoid H(w*I_net + ...). 
    # H is nonlinear. So coupling C_12*S_2 will generate terms involving S_2, but also S_2*S_1 etc. via Taylor expansion.
    # A pure linear C_ij might not appear as just "C * S_2" if H(S) is highly nonlinear.
    # BUT, if we look for dependence on S2 in dS1, that confirms connectivity recovery.
    
    # Check if dS1 depends on S2 (idx 2 in library)
    coupling_1_from_2 = jnp.abs(Xi[2, 0]) > 1e-4 # S2 term in eq 1
    # Check if dS2 depends on S1 (idx 1 in library)
    coupling_2_from_1 = jnp.abs(Xi[1, 1]) > 1e-4 # S1 term in eq 2
    
    print(f"Node 1 detected input from Node 2? {coupling_1_from_2}")
    print(f"Node 2 detected input from Node 1? {coupling_2_from_1}")
    
    return Xi

if __name__ == "__main__":
    demo_sindy_network()
