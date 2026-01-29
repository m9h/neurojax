
import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
import matplotlib.pyplot as plt
import numpy as np

# Add examples to path to import loader and model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sleep_flow_data_loader import load_and_process_data
from sleep_flow_model import NeuralODE, VectorField

def main():
    # 1. Load Data
    print("Loading data...")
    # Expects data to be downloaded in downloads/ds003768
    data, mean, std = load_and_process_data(subject_id='sub-01', data_dir='downloads/ds003768')
    # data shape: (n_steps, 5)
    
    # 2. Setup Training
    ts = jnp.arange(data.shape[0]) * 30.0 # 30s epochs (assuming independent or sliding steps)
    # Actually, the loader does sliding window 5s steps?
    # sleep_flow_data_loader.py: window 30s, step 5s.
    # So dt is 5.0 seconds.
    dt_step = 5.0
    ts = jnp.arange(data.shape[0]) * dt_step
    
    key = jax.random.PRNGKey(42)
    model = NeuralODE(key)
    
    # Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Loss Function: One-step or Multi-step prediction
    # For stability, let's predict next step or short chunk
    # "Multiple Shooting" approach: Predict next k steps from current step.
    
    @eqx.filter_value_and_grad
    def loss_fn(model, batch_t, batch_y):
        # batch_y: (batch_size, seq_len, 5)
        # batch_t: (batch_size, seq_len)
        
        # We predict trajectory from t[0] to t[-1] starting at y[0]
        # And compare with y[1:]
        
        preds = jax.vmap(lambda y0, t: model(y0, t))(batch_y[:, 0, :], batch_t)
        
        # Preds is the full trajectory at times t
        # MSE against ground truth
        return jnp.mean((preds - batch_y) ** 2)

    # Batching
    def get_batch(data, dt_step, batch_size=32, seq_len=10, key=None):
        # Sample random start indices
        max_idx = data.shape[0] - seq_len
        idxs = jax.random.randint(key, (batch_size,), 0, max_idx)
        
        # Vectorized slice
        # Create indices: (batch, seq)
        b_idxs = idxs[:, None] + jnp.arange(seq_len)[None, :]
        batch_y = data[b_idxs]
        
        # Time is always relative 0, dt, 2dt... for autonomous flow check
        batch_t = jnp.tile(jnp.arange(seq_len) * dt_step, (batch_size, 1))
            
        return batch_y, batch_t

    # Training Loop
    n_epochs = 1000
    print("Training...")
    
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        batch_y, batch_t = get_batch(data, dt_step, batch_size=64, seq_len=5, key=subkey)
        
        loss, grads = loss_fn(model, batch_t, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
    # 3. Visualization
    print("Generating visualizations...")
    
    # A. Streamplot in Delta-Alpha plane (assuming idx 0 and 2 for Delta, Alpha)
    # Bands: Delta(0), Theta(1), Alpha(2), Sigma(3), Beta(4)
    # We fix other dims to 0 (mean) for the slice.
    
    delta_range = jnp.linspace(-3, 3, 20)
    alpha_range = jnp.linspace(-3, 3, 20)
    D, A = jnp.meshgrid(delta_range, alpha_range)
    
    # Create grid points: (D, 0, A, 0, 0)
    grid_points = jnp.zeros((D.size, 5))
    grid_points = grid_points.at[:, 0].set(D.ravel())
    grid_points = grid_points.at[:, 2].set(A.ravel())
    
    # Compute vector field
    # VectorField is in model.vector_field
    # f(t, y, args)
    vf = jax.vmap(lambda y: model.vector_field(0, y, None))(grid_points)
    
    U = np.array(vf[:, 0].reshape(D.shape)) # d(Delta)/dt
    V = np.array(vf[:, 2].reshape(D.shape)) # d(Alpha)/dt
    D = np.array(D)
    A = np.array(A)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    st = ax.streamplot(D, A, U, V, color=np.sqrt(U**2 + V**2), cmap='viridis')
    ax.set_xlabel('Delta Power (z-score)')
    ax.set_ylabel('Alpha Power (z-score)')
    ax.set_title('Sleep Flow Vector Field (Delta vs Alpha)')
    plt.colorbar(st.lines, label='Flow Speed')
    
    # Overlay actual trajectory (first 1000 points) to see drift
    ax.plot(data[:1000, 0], data[:1000, 2], 'k-', alpha=0.3, label='Actual Trajectory (Early)')
    ax.plot(data[-1000:, 0], data[-1000:, 2], 'r-', alpha=0.3, label='Actual Trajectory (Late)')
    ax.legend()
    
    plt.savefig('sleep_flow_streamplot.png')
    print("Saved sleep_flow_streamplot.png")
    
    # B. Homeostatic Drift
    # Visualize the flow vector magnitude or direction over the session time
    # Or just plotting the trajectory in 3D (Delta, Alpha, Time) or just (Delta, Alpha) color-coded by time
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sc = ax2.scatter(data[:, 0], data[:, 2], c=ts, cmap='plasma', s=2, alpha=0.5)
    ax2.set_xlabel('Delta Power (z-score)')
    ax2.set_ylabel('Alpha Power (z-score)')
    ax2.set_title('Trajectory with Time Drift')
    plt.colorbar(sc, label='Time (s)')
    plt.savefig('sleep_flow_drift.png')
    print("Saved sleep_flow_drift.png")
    
if __name__ == "__main__":
    main()
