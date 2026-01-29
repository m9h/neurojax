
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import pandas as pd
from neurojax.dynamics import SINDyOptimizer

def drug_library(X_combined):
    """
    Custom SINDy library including:
    - Linear state and control: x, u
    - Linear Interaction: x * u
    - Saturating Interaction: x * (u / (1.0 + u))
    """
    # Expect X_combined to be (T, 2) where col 0 is x, col 1 is u
    x = X_combined[:, 0:1]
    u = X_combined[:, 1:2]
    
    # Constants
    # 1. Linear Terms
    # x, u are already there.
    
    # 2. Interaction Terms
    x_u_linear = x * u
    
    # 3. Saturating Terms (Michaelis-Menten with Km=1.0)
    # Testing the hypothesis that effect saturates
    x_u_sat = x * (u / (1.0 + u))
    
    # 4. Bias/Intercept
    bias = jnp.ones_like(x)
    
    return jnp.concatenate([bias, x, u, x_u_linear, x_u_sat], axis=1)

def recover_dose_response():
    print("--- 1. CONFIGURATION & SYNTHETIC DATA GENERATION ---")
    
    # Pharmacokinetics (PK)
    # 1-compartment model: u(t) = D0 * exp(-k * t)
    t_half = 2.0 # hours
    k_elim = np.log(2) / t_half
    D0 = 10.0 # High dose (relative to generic EC50)
    
    # Pharmacodynamics (PD)
    # Ground Truth: dx/dt = A*x - Emax * (u / (EC50 + u)) * x
    # We choose EC50 = 1.0 to match our testing library (which tests K=1.0)
    # If we wanted to "estimate" EC50 generally, we'd need a parameterized library scan.
    # For this demo, we verify we can distinguish Linear vs Saturating form.
    
    A_true = -0.5 # Intrinsic decay/damping of the neural feature without drug
    I_drive = 2.0 # Constant drive to maintain baseline activity > 0
    Emax_true = 2.0
    EC50_true = 1.0 
    
    print(f"Ground Truth Parameters:")
    print(f"  PK Half-life: {t_half} h")
    print(f"  Intrinsic Decay A: {A_true}")
    print(f"  Base Drive I: {I_drive}")
    print(f"  Drug Effect Emax: {Emax_true}")
    print(f"  Drug Potency EC50: {EC50_true}")
    
    # Simulation Setup
    t0 = 0.0
    t1 = 24.0 # 24 hours (12 half-lives) to ensure washout and identifiability
    dt = 0.01
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    
    # Define vector field for generation
    def vector_field(t, y, args):
        x = y[0]
        # Explicit PK
        u = D0 * jnp.exp(-k_elim * t)
        
        # PD Dynamics
        # Effect is inhibition/excitation. Let's assume it increases decay (anticonvulsant suppressing activity)
        # or drives it. Let's say it suppresses: dx = Ax - Effect*x
        # Wait, if A is negative (-0.5), it's already decaying. 
        # Let's make the drug DRIVE the state or MODULATE it.
        # "Anticonvulsant" -> Suppresses excitability.
        # Let's assume x is "Epileptiform Activity". 
        # Autonomous: dx = +0.5x (Unstable/Growing) 
        # Drug: -Emax * ... * x (Stabilizing)
        # Result: Stable with drug, unstable without?
        # Or simply: dx = -0.5x (Stable baseline) + Drug Effect?
        # Let's stick to the prompt's implied form: dx = A x + B(u) x.
        # Let's set A = -0.1 (slow decay), and drug adds MORE decay (B(u) is negative).
        
        # Using the prompt's notation: dx = Ax + B(u)x + I
        # B(u) = - Emax * u / (EC50 + u)
        
        dxdt = A_true * x + I_drive - Emax_true * (u / (EC50_true + u)) * x
        return jnp.array([dxdt])

    # Initial Condition
    y0 = jnp.array([1.0]) # Start near suppressed state? Or random.
    
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0, t1, dt, y0, saveat=saveat
    )
    
    X_syn = sol.ys # Shape (N, 1)
    Times = sol.ts
    
    # Compute u(t) corresponding to these times for the SINDy library
    U_syn = D0 * jnp.exp(-k_elim * Times)[:, None] # Shape (N, 1)
    
    # Combine for library input
    X_combined = jnp.concatenate([X_syn, U_syn], axis=1) # (N, 2)
    
    # Calculate exact derivatives for SINDy target
    # (In practice we'd derivative the data, but for demo validation use exact to isolate SINDy perf)
    dX_syn = jax.vmap(lambda t, y: vector_field(t, y, None))(Times, X_syn)
    
    print(f"Simulated {len(Times)} timepoints.")
    
    
    print("\n--- 2. SINDY WITH CONTROL RECOVERY ---")
    
    # Optimizer
    optimizer = SINDyOptimizer(threshold=0.1) # Threshold to kill small coeffs
    
    print("Fitting SINDy model...")
    # fit(X, dX, library)
    # X here is the input to the library (state + control)
    # dX is the target derivative (only state derivative, shape (N,1))
    
    Xi = optimizer.fit(X_combined, dX_syn, drug_library)
    
    feature_names = ["1", "x", "u", "x*u", "x*u/(1+u)"]
    # library columns:
    # 0: 1
    # 1: x
    # 2: u
    # 3: x*u
    # 4: x*u/(1+u)
    
    print("\n--- Discovered Coefficients (Target: dx/dt) ---")
    coeffs = Xi[:, 0] # First column of Xi corresponds to the first state variable derivative
    
    for name, val in zip(feature_names, coeffs):
        print(f"{name:>12}: {val:.4f}")

    print("\n--- 3. VERIFICATION & ANALYSIS ---")
    
    # Check 1: Is Linear Interaction (x*u) zero/sparse?
    coeff_linear_int = coeffs[3]
    is_linear_rejected = abs(coeff_linear_int) < 0.1
    
    # Check 2: Is Saturating Interaction (x*u/(1+u)) selected?
    coeff_sat_int = coeffs[4]
    is_sat_selected = abs(coeff_sat_int) > 0.1
    
    print(f"Hypothesis Test: Linear Interaction Rejected? {is_linear_rejected}")
    print(f"Hypothesis Test: Saturating Interaction Selected? {is_sat_selected}")
    
    if is_sat_selected and is_linear_rejected:
        print("SUCCESS: Sparse Regression correctly identified Saturating Dynamics.")
        
        # Recovered Emax is the negative of the coefficient for saturation term (since we modeled -Emax)
        # Truth: -Emax * ...
        # Coeff: should be approx -Emax
        Emax_recovered = -coeff_sat_int
        
        # EC50 Estimation
        # Since our library term was exactly u/(1+u), we implicitly assumed Km=1.
        # If the fit is good with this term, the estimated EC50 is dominated by our library choice (1.0).
        # In a real scenario, we would fit u/(K+u) for varying K and pick the best fit (SINDy-PI style).
        # For this demo, we report the Km of the selected feature.
        EC50_estimated = 1.0 
        
        print(f"\nRecovered Parameters:")
        print(f"  Emax: {Emax_recovered:.4f} (True: {Emax_true})")
        print(f"  EC50: {EC50_estimated:.4f} (True: {EC50_true})") # By construct
        
        # Verify Accuracy
        err = abs(Emax_recovered - Emax_true) / Emax_true
        if err < 0.1:
            print("Parameter recovery within 10% tolerance.")
        else:
            print("Parameter recovery deviating.")
            
    else:
        print("FAILURE: SINDy failed to distinguish the correct model structure.")

if __name__ == "__main__":
    recover_dose_response()
