import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, r=4.0):
    return r * x * (1.0 - x)

def sigmoid(x):
    return 1 / (1 + np.exp(-10 * (x - 0.5)))

def GWO(fobj, lb, ub, dim, SearchAgents_no, Max_iter, variant='original'):
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')

    # Handle lb, ub if scalar
    if not isinstance(lb, (list, np.ndarray)): lb = np.full(dim, lb)
    if not isinstance(ub, (list, np.ndarray)): ub = np.full(dim, ub)

    # --- Initialize according to variant ---
    if variant == 'binary':
        # BGWO requires initial positions as 0 or 1
        Positions = np.random.randint(0, 2, (SearchAgents_no, dim))
    else:
        # Continuous variants
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    
    # Initialize for Hybrid GWO-PSO
    if variant == 'hybrid_pso':
        Velocities = np.zeros((SearchAgents_no, dim))
        Pbest_pos = Positions.copy()
        Pbest_score = np.full(SearchAgents_no, float('inf'))
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter

    Convergence_curve = np.zeros(Max_iter)
    
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            
            if variant != 'binary':
                Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            
            fitness = fobj(Positions[i, :])
            
            if variant == 'hybrid_pso':
                if fitness < Pbest_score[i]:
                    Pbest_score[i] = fitness
                    Pbest_pos[i, :] = Positions[i, :].copy()
            if fitness < Alpha_score:
                Alpha_score, Beta_score, Delta_score = fitness, Alpha_score, Beta_score
                Alpha_pos, Beta_pos, Delta_pos = Positions[i, :].copy(), Alpha_pos.copy(), Beta_pos.copy()
            elif Alpha_score < fitness < Beta_score:
                Beta_score, Delta_score = fitness, Beta_score
                Beta_pos, Delta_pos = Positions[i, :].copy(), Beta_pos.copy()
            elif Alpha_score < fitness and Beta_score < fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        if variant == 'igwo':
            a = 2 * (1 - (t / Max_iter) ** 2)
        elif variant == 'mogwo':
            a = 2 - t * (2 / Max_iter) * (1 + 0.5 * np.sin(2 * np.pi * t / Max_iter))
        else:
            a = 2 - t * (2 / Max_iter)
        r1_alpha, r2_alpha = np.random.rand(SearchAgents_no, dim), np.random.rand(SearchAgents_no, dim)
        r1_beta, r2_beta = np.random.rand(SearchAgents_no, dim), np.random.rand(SearchAgents_no, dim)
        r1_delta, r2_delta = np.random.rand(SearchAgents_no, dim), np.random.rand(SearchAgents_no, dim)

        A1, C1 = 2 * a * r1_alpha - a, 2 * r2_alpha
        D_alpha = np.abs(C1 * Alpha_pos - Positions)
        X1 = Alpha_pos - A1 * D_alpha

        A2, C2 = 2 * a * r1_beta - a, 2 * r2_beta
        D_beta = np.abs(C2 * Beta_pos - Positions)
        X2 = Beta_pos - A2 * D_beta

        A3, C3 = 2 * a * r1_delta - a, 2 * r2_delta
        D_delta = np.abs(C3 * Delta_pos - Positions)
        X3 = Delta_pos - A3 * D_delta

        if variant == 'hybrid_pso':
            gwo_pos = (X1 + X2 + X3) / 3
            r1_pso = np.random.rand(SearchAgents_no, dim)
            r2_pso = np.random.rand(SearchAgents_no, dim)
            Velocities = w * Velocities + c1 * r1_pso * (Pbest_pos - Positions) + c2 * r2_pso * (Alpha_pos - Positions)
            continuous_pos = 0.5 * gwo_pos + 0.5 * (Positions + Velocities)
        else:
            continuous_pos = (X1 + X2 + X3) / 3

        if variant == 'binary':
            probabilities = sigmoid(continuous_pos)
            Positions = np.where(np.random.rand(SearchAgents_no, dim) < probabilities, 1, 0)
        else:
            Positions = continuous_pos

        Convergence_curve[t] = Alpha_score
        
    return Alpha_score, Alpha_pos, Convergence_curve

if __name__ == "__main__":
    
    def rastrigin_func(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    SearchAgents_no = 30
    Max_iter = 500
    dim_cont = 10
    lb_cont = -5.12
    ub_cont = 5.12
    
    variants_to_run = [
        'original',
        'igwo',
        'mogwo',
        'hybrid_pso'
    ]
    
    results = {}

    print("Starting comparison of GWO variants on Rastrigin function...")
    print(f"Problem: Rastrigin ({dim_cont}D), Range: [{lb_cont}, {ub_cont}]")
    print(f"Population: {SearchAgents_no}, Iterations: {Max_iter}\n")

    for variant in variants_to_run:
        print(f"  Running: {variant.upper()}...")
        fobj = rastrigin_func
        score, pos, curve = GWO(fobj, lb_cont, ub_cont, dim_cont, SearchAgents_no, Max_iter, variant=variant)
        results[variant] = {'score': score, 'curve': curve}
        print(f"  Completed. Best Score: {score:.6e}\n")

    plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for idx, (variant, data) in enumerate(results.items()):
        plt.plot(data['curve'], 
                label=variant.upper(), 
                lw=2.5, 
                color=colors[idx % len(colors)])
    
    plt.title('Convergence Curve Comparison of GWO Variants (Rastrigin Function)', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness Value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("Starting Binary variant (BGWO)...")
    
    def one_max_func(x):
        return -np.sum(x)

    dim_bin = 40
    
    score_bin, pos_bin, curve_bin = GWO(
        one_max_func, 0, 1, dim_bin, SearchAgents_no, Max_iter, variant='binary'
    )
    
    print(f"  Completed BGWO. Best Score: {score_bin}")
    print(f"  (Found {int(-score_bin)} ones out of {dim_bin} bits)")

    # Plot for BGWO
    plt.figure(figsize=(12, 7))
    plt.plot(curve_bin, label=f'BGWO (OneMax - {dim_bin} bits)', color='darkred', lw=2.5)
    plt.title('Convergence Curve of Binary GWO (BGWO) - OneMax Problem', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Variant':<15} {'Best Fitness':<20} {'Final Convergence':<20}")
    print("-"*60)
    for variant, data in results.items():
        print(f"{variant.upper():<15} {data['score']:<20.6e} {data['curve'][-1]:<20.6e}")
    print(f"{'BGWO':<15} {score_bin:<20.6e} {curve_bin[-1]:<20.6e}")
    print("="*60)