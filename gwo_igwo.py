import numpy as np
import matplotlib.pyplot as plt

def GWO(fobj, lb, ub, dim, SearchAgents_no, Max_iter, variant='original'):
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')

    if not isinstance(lb, (list, np.ndarray)): lb = np.full(dim, lb)
    if not isinstance(ub, (list, np.ndarray)): ub = np.full(dim, ub)

    Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb

    Convergence_curve = np.zeros(Max_iter)
    
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])
            
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

        Positions = (X1 + X2 + X3) / 3

        Convergence_curve[t] = Alpha_score
        
    return Alpha_score, Alpha_pos, Convergence_curve

if __name__ == "__main__":
    
    def sphere_func(x):
        return np.sum(x**2)
    
    SearchAgents_no = 30
    Max_iter = 500
    dim_cont = 10
    lb_cont = -100
    ub_cont = 100
    
    variants_to_run = ['original', 'igwo']
    results = {}

    print("Starting comparison of GWO and IGWO on Sphere function...")
    print(f"Problem: Sphere ({dim_cont}D), Range: [{lb_cont}, {ub_cont}]")
    print(f"Population: {SearchAgents_no}, Iterations: {Max_iter}\n")

    for variant in variants_to_run:
        print(f"  Running: {variant.upper()}...")
        score, pos, curve = GWO(sphere_func, lb_cont, ub_cont, dim_cont, SearchAgents_no, Max_iter, variant=variant)
        results[variant] = {'score': score, 'curve': curve}
        print(f"  Completed. Best Score: {score:.6e}\n")

    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'green']
    
    for idx, (variant, data) in enumerate(results.items()):
        plt.plot(data['curve'], 
                label=variant.upper(), 
                lw=2.5, 
                color=colors[idx])
    
    plt.title('So sánh đường cong hội tụ: GWO vs IGWO (Σ(xi²))', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness Value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Variant':<15} {'Best Fitness':<20} {'Final Convergence':<20}")
    print("-"*60)
    for variant, data in results.items():
        print(f"{variant.upper():<15} {data['score']:<20.6e} {data['curve'][-1]:<20.6e}")
    print("="*60)
