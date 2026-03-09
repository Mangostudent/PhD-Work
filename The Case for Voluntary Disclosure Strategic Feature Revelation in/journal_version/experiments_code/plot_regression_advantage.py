import numpy as np
import matplotlib.pyplot as plt
import os
import time

def run_experiment():
    print("Computing regression strategic advantage heatmap (Optimized)...")
    t0 = time.time()
    
    q_vals = np.linspace(0.01, 0.99, 100)
    cov_yu_vals = np.linspace(-0.95, 0.95, 100)
    b_vals = np.linspace(-3, 3, 100)
    
    n_mc = 200000
    
    # Base standard normals
    np.random.seed(42)
    Z1 = np.random.randn(n_mc)
    Z2 = np.random.randn(n_mc)
    Z3 = np.random.randn(n_mc)
    
    # X and U have Cov(X,U) = 0.2
    # X = Z1
    # U = 0.2*Z1 + sqrt(1 - 0.2^2)*Z2
    X = Z1
    U = 0.2 * Z1 + np.sqrt(1 - 0.04) * Z2
    
    advantage = np.zeros((len(q_vals), len(cov_yu_vals)))
    
    b_vals = b_vals.reshape(1, -1)
    X_mat = X.reshape(-1, 1)
    U_mat = U.reshape(-1, 1)
    
    bX = b_vals * X_mat # (n_mc, len(b))
    F = (2 * U_mat - bX) * bX
    mask = F > 0 # (n_mc, len(b))
    
    var_X = 1.0
    cov_xy = 0.2
    b_Y = cov_xy / var_X # 0.2
    
    for j, cov_yu in enumerate(cov_yu_vals):
        # We need Y such that Var(Y)=1, Cov(X,Y)=0.2, Cov(U,Y)=cov_yu
        # Let Y = a*X + b*U + c*Z3
        # Cov(X,Y) = a*Var(X) + b*Cov(X,U) = a + 0.2*b = 0.2
        # Cov(U,Y) = a*Cov(X,U) + b*Var(U) = 0.2*a + b = cov_yu
        # Solving for a and b:
        # a = (0.2 - 0.2*cov_yu) / (1 - 0.04)
        # b = (cov_yu - 0.04) / (1 - 0.04)
        a = (0.2 - 0.2*cov_yu) / 0.96
        b_coef = (cov_yu - 0.04) / 0.96
        
        # Var(Y) = a^2 + b^2 + 2ab(0.2) + c^2 = 1
        # c^2 = 1 - (a^2 + b^2 + 0.4*a*b)
        c2 = 1 - (a**2 + b_coef**2 + 0.4*a*b_coef)
        if c2 < 0:
            advantage[:, j] = np.nan
            continue
            
        c = np.sqrt(c2)
        
        Y = a * X + b_coef * U + c * Z3
        Y_mat = Y.reshape(-1, 1) # (n_mc, 1)
        
        # Compute G
        G = (2 * Y_mat - bX) * bX # (n_mc, len(b))
        
        # Expected value
        EG_mask = np.mean(G * mask, axis=0) # (len(b),)
        
        # Rs(b, q) = Var(Y) - (1-q) * EG_mask
        # min_b Rs(b, q) = 1 - (1-q) * max_b EG_mask
        max_EG = np.max(EG_mask)
        Rs_opt = 1.0 - (1 - q_vals) * max_EG # array over q
        
        # Rv(q) = Var(Y) - (1-q) * b_Y^2 * Var(X) = 1 - (1-q) * 0.04
        Rv_q = 1.0 - (1 - q_vals) * 0.04
        
        advantage[:, j] = Rv_q - Rs_opt
        
        if (j + 1) % 10 == 0:
            print(f"  Col {j+1}/{len(cov_yu_vals)} done in {time.time()-t0:.1f}s")
            
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = max(abs(np.nanmin(advantage)), abs(np.nanmax(advantage)))
    im = ax.imshow(
        advantage, origin='lower', aspect='auto',
        extent=[cov_yu_vals[0], cov_yu_vals[-1], q_vals[0], q_vals[-1]],
        cmap='RdBu', vmin=-vmax, vmax=vmax,
        interpolation='bilinear'
    )
    
    # Draw theoretical condition line
    ax.axvline(x=0.88, color='black', linestyle=':', linewidth=2.5, label='If and Only If Condition')
    
    ax.set_xlabel(r"Preference Alignment $Cov(Y, U)$", fontsize=14)
    ax.set_ylabel(r"Noisy Channel Probability $q$", fontsize=14)
    ax.set_title("Strategic Advantage for the Regression Problem", fontsize=16, pad=15)
    ax.legend(fontsize=12, loc='upper left')
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Strategic Advantage (Vanilla Risk - Strategic Risk)", fontsize=12, rotation=270, labelpad=20)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, '..', 'figures', 'regression_strategic_advantage.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    run_experiment()
