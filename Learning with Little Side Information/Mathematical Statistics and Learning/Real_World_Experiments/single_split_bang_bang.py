import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def get_real_dataset(name):
    print(f"Loading {name}...")
    if name == 'adult':
        data = fetch_openml(data_id=1590, as_frame=True)
        df = data.frame.dropna(subset=['class', 'sex'])
        Y_raw = df['class']
        Z_raw = df['sex']
        X_df = df.drop(columns=['class', 'sex'])
    elif name == 'bank':
        data = fetch_openml(data_id=1461, as_frame=True)
        df = data.frame.dropna(subset=['Class', 'V7'])
        Y_raw = df['Class']
        Z_raw = df['V7'] 
        X_df = df.drop(columns=['Class', 'V7'])
    elif name == 'credit-g':
        data = fetch_openml(data_id=31, as_frame=True)
        df = data.frame.dropna(subset=['class', 'foreign_worker'])
        Y_raw = df['class']
        Z_raw = df['foreign_worker']
        X_df = df.drop(columns=['class', 'foreign_worker'])
    elif name == 'titanic':
        data = fetch_openml(data_id=40945, as_frame=True)
        df = data.frame.dropna(subset=['survived', 'sex', 'pclass', 'age', 'fare'])
        Y_raw = df['survived']
        Z_raw = df['sex']
        X_df = df[['pclass', 'age', 'fare']]
    elif name == 'breast-w':
        data = fetch_openml(data_id=15, as_frame=True)
        df = data.frame.dropna(subset=['Class', 'Clump_Thickness'])
        Y_raw = df['Class']
        Z_raw = df['Clump_Thickness'] > 5
        X_df = df.drop(columns=['Class', 'Clump_Thickness'])
    elif name == 'diabetes':
        data = fetch_openml(data_id=37, as_frame=True)
        df = data.frame.dropna(subset=['class', 'age'])
        Y_raw = df['class']
        Z_raw = df['age'] > 30
        X_df = df.drop(columns=['class', 'age'])
    elif name == 'spambase':
        data = fetch_openml(data_id=44, as_frame=True)
        df = data.frame.dropna(subset=['class', 'word_freq_make'])
        Y_raw = df['class']
        Z_raw = df['word_freq_make'] > 0
        X_df = df.drop(columns=['class', 'word_freq_make'])
    elif name == 'sick':
        data = fetch_openml(data_id=38, as_frame=True)
        df = data.frame.dropna(subset=['Class', 'sex'])
        Y_raw = df['Class']
        Z_raw = df['sex']
        X_df = df.drop(columns=['Class', 'sex'])
    elif name == 'mushroom':
        data = fetch_openml(data_id=24, as_frame=True)
        df = data.frame.dropna(subset=['class', 'bruises%3F'])
        Y_raw = df['class']
        Z_raw = df['bruises%3F']
        X_df = df.drop(columns=['class', 'bruises%3F'])
    else:
        raise ValueError(f"Dataset {name} not found")

    Y = np.where(Y_raw == Y_raw.iloc[0], 1.0, -1.0)
    Z = np.where(Z_raw == Z_raw.iloc[0], 1.0, -1.0)
    
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_df.select_dtypes(exclude=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    X = preprocessor.fit_transform(X_df)
    return X, Y, Z

def _loss_function(w, X_full, Y, lambda_c, lambda_x, lambda_z):
    scores = np.dot(X_full, w)
    margin = Y * scores
    loss_term = np.sum(-np.log(sigmoid(margin) + 1e-15))
    probs = sigmoid(-margin)
    grad_loss = -np.dot(X_full.T, probs * Y)
    
    reg_term = (lambda_c * w[0]**2 +
                lambda_x * np.sum(w[1:-1]**2) +
                lambda_z * w[-1]**2)
    
    grad_reg = w.copy()
    grad_reg[0] *= 2 * lambda_c
    grad_reg[1:-1] *= 2 * lambda_x
    grad_reg[-1] *= 2 * lambda_z

    return loss_term + reg_term, grad_loss + grad_reg

def find_optimal_lambda(X, Y, Z, lambda_z_grid, seed):
    # Split of the real-world dataset
    X_train, X_pop, Y_train, Y_pop, Z_train, Z_pop = train_test_split(
        X, Y, Z, test_size=0.3, random_state=seed
    )
    X_d, X_s, Y_d, Y_s, Z_d, Z_s = train_test_split(
        X_train, Y_train, Z_train, test_size=0.21, random_state=seed
    )
    
    phi_model = LogisticRegression(C=1.0, solver='lbfgs', random_state=seed, max_iter=500)
    # Ensure at least 2 classes in Z_s
    if len(np.unique(Z_s)) < 2:
        return np.random.choice(lambda_z_grid)
        
    phi_model.fit(X_s, Z_s)
    probs_z = phi_model.predict_proba(X_d)[:, 1]
    Z_d_hat = 2.0 * (probs_z >= 0.5) - 1.0

    lambda_c = 1.0 / np.sqrt(X_d.shape[0])
    lambda_x = 1.0 / np.sqrt(X_d.shape[0])

    X_full_d = np.hstack([np.ones((X_d.shape[0], 1)), X_d, Z_d_hat.reshape(-1, 1)])
    X_full_pop = np.hstack([np.ones((X_pop.shape[0], 1)), X_pop, Z_pop.reshape(-1, 1)])

    best_risk = float('inf')
    best_lam = None
    
    for lam_z in lambda_z_grid:
        w_init = np.zeros(X_full_d.shape[1])
        res = minimize(_loss_function, x0=w_init, args=(X_full_d, Y_d, lambda_c, lambda_x, lam_z), method='L-BFGS-B', jac=True)
        pop_loss, _ = _loss_function(res.x, X_full_pop, Y_pop, 0, 0, 0)
        pop_risk = pop_loss / X_pop.shape[0]

        if pop_risk < best_risk:
            best_risk = pop_risk
            best_lam = lam_z
            
    return best_lam

if __name__ == "__main__":
    datasets = ['adult', 'bank', 'credit-g', 'titanic', 'breast-w', 'diabetes', 'spambase', 'sick', 'mushroom']
    lambda_z_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    results = {}
    
    for ds in datasets:
        try:
            X, Y, Z = get_real_dataset(ds)
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
            
            best_lam = find_optimal_lambda(X, Y, Z, lambda_z_grid, seed=42)
            results[ds] = best_lam
            
            print(f">>> {ds.upper()} optimal lambda_z = {best_lam} <<<\n")
        except Exception as e:
            print(f"Failed {ds}: {e}")

    # Sort by optimal lambda value (low to high)
    sorted_items = sorted(results.items(), key=lambda item: item[1])
    sorted_keys = [item[0] for item in sorted_items]
    sorted_values = [item[1] for item in sorted_items]

    # Plot
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in np.linspace(0, 1, len(sorted_keys))]
    
    bars = plt.bar(sorted_keys, sorted_values, color=colors, edgecolor='black')
    
    plt.yscale('log')
    plt.ylabel(r"Optimal $\lambda_z^*$ (Log Scale)", fontsize=14)
    plt.title(r"Optimal Regularization $\lambda_z^*$ on 9 Real-World Datasets", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Text on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.5, f'{height}', ha='center', va='bottom', fontweight='bold')
        
    plt.grid(axis='y', alpha=0.3)
    
    # Increase ylim to accommodate text labels on tallest bars
    plt.ylim(min(lambda_z_grid) / 2, max(lambda_z_grid) * 10)
    
    plt.tight_layout()
    output_path = "optimal_lambda_real_world.png"
    plt.savefig(output_path, dpi=300)
    print(f"Simulation complete! Saved plot to {output_path}")
