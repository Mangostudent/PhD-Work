import numpy as np
import matplotlib.pyplot as plt
import os
import time

def run_experiment():
    print("Computing classification strategic advantage heatmap (Optimized)...")
    t0 = time.time()
    
    corr_vals = np.linspace(-0.99, 0.99, 100)
    q_vals = np.linspace(0.01, 0.99, 100)
    
    advantage = np.zeros((len(q_vals), len(corr_vals)))
    
    n_mc = 200000
    np.random.seed(42)
    Z = np.random.randn(n_mc)
    rand_unif = np.random.rand(n_mc)
    
    p_Y1 = 0.6
    p_U1 = 0.5
    var_Y = 1.0 - (2*p_Y1 - 1)**2
    var_U = 1.0
    
    mu_Y, mu_U = 1.5, 0.5
    
    for j, corr_yu in enumerate(corr_vals):
        cov_yu = corr_yu * np.sqrt(var_Y * var_U)
        a = (cov_yu + 2*(p_Y1 + p_U1) - 1) / 4
        
        probs = np.array([
            a,                   # Y=1, U=1
            p_Y1 - a,           # Y=1, U=-1
            p_U1 - a,           # Y=-1, U=1
            1 - p_Y1 - p_U1 + a # Y=-1, U=-1
        ])
        
        if np.any(probs < -1e-8):
            advantage[:, j] = np.nan
            continue
            
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        
        cum_probs = np.cumsum(probs)
        cat = np.searchsorted(cum_probs, rand_unif)
        
        Y = np.where((cat == 0) | (cat == 1), 1, -1)
        U = np.where((cat == 0) | (cat == 2), 1, -1)
        X = Y * mu_Y + U * mu_U + Z
        
        def px_yu(y, u):
            # Proportional to likelihood Exp(-0.5 * (X - mu)^2)
            # We can just compute log likelihood
            return -0.5 * (X - (y*mu_Y + u*mu_U))**2
            
        l11 = px_yu(1, 1) + np.log(max(probs[0], 1e-30))
        l1m1 = px_yu(1, -1) + np.log(max(probs[1], 1e-30))
        lm11 = px_yu(-1, 1) + np.log(max(probs[2], 1e-30))
        lm1m1 = px_yu(-1, -1) + np.log(max(probs[3], 1e-30))
        
        # eta(X)
        l_Y1 = np.logaddexp(l11, l1m1)
        l_Ym1 = np.logaddexp(lm11, lm1m1)
        l_d = np.logaddexp(l_Y1, l_Ym1)
        eta_X = 2 * np.exp(l_Y1 - l_d) - 1
        
        R_V_base_1 = np.mean((1 - np.abs(eta_X)) / 2)
        # RV*(q) = q*(1-0.6) + (1-q)*R_V_base_1
        R_V = q_vals * 0.4 + (1 - q_vals) * R_V_base_1
        
        # eta_given_u
        # For U=1
        l_d_U1 = np.logaddexp(l11, lm11)
        eta_p1 = 2 * np.exp(l11 - l_d_U1) - 1
        
        # For U=-1
        l_d_Um1 = np.logaddexp(l1m1, lm1m1)
        eta_m1 = 2 * np.exp(l1m1 - l_d_Um1) - 1
        
        R_base_m1 = np.mean((1 - np.abs(eta_m1)) / 2)
        R_base_p1 = np.mean((1 - np.abs(eta_p1)) / 2)
        
        pi_1, pi_m1 = p_U1, 1 - p_U1
        p_Ym1_U1 = probs[2] / max(p_U1, 1e-30)
        p_Ym1_Um1 = probs[3] / max(1 - p_U1, 1e-30)
        p_Y1_U1 = probs[0] / max(p_U1, 1e-30)
        p_Y1_Um1 = probs[1] / max(1 - p_U1, 1e-30)
        
        R_S1 = pi_1 * p_Ym1_U1 + pi_m1 * ((1 - q_vals) * R_base_m1 + q_vals * p_Ym1_Um1)
        R_Sm1 = pi_m1 * p_Y1_Um1 + pi_1 * ((1 - q_vals) * R_base_p1 + q_vals * p_Y1_U1)
        
        R_S = np.minimum(R_S1, R_Sm1)
        
        advantage[:, j] = R_V - R_S
        
        if (j + 1) % 10 == 0:
            print(f"  Col {j+1}/{len(corr_vals)} done in {time.time()-t0:.1f}s")
            
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = max(abs(np.nanmin(advantage)), abs(np.nanmax(advantage)))
    im = ax.imshow(
        advantage, origin='lower', aspect='auto',
        extent=[corr_vals[0], corr_vals[-1], q_vals[0], q_vals[-1]],
        cmap='RdBu', vmin=-vmax, vmax=vmax,
        interpolation='bilinear'
    )
    
    ax.axvline(x=0.85, color='black', linestyle=':', linewidth=2.5, label='Theoretical Condition')
    
    ax.set_xlabel(r"Preference Alignment $Corr(Y, U)$", fontsize=14)
    ax.set_ylabel(r"Noisy Channel Probability $q$", fontsize=14)
    ax.set_title("Strategic Advantage for the Classification Problem", fontsize=16, pad=15)
    ax.legend(fontsize=12, loc='upper left')
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Strategic Advantage (Vanilla Risk - Strategic Risk)", fontsize=12, rotation=270, labelpad=20)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, '..', 'figures', 'classification_strategic_advantage.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    run_experiment()
