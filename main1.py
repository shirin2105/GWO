import numpy as np
import matplotlib.pyplot as plt

def GWO(fobj, lb, ub, dim, SearchAgents_no, Max_iter, variant='original'):
    # Khởi tạo tham số
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
        # Cập nhật fitness và các vị trí lãnh đạo
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])
            
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        a = 2 - t * (2 / Max_iter)

        for i in range(SearchAgents_no):
            # Tính toán vị trí theo GWO chuẩn
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * Alpha_pos - Positions[i, :])
            X1 = Alpha_pos - A1 * D_alpha
            
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * Beta_pos - Positions[i, :])
            X2 = Beta_pos - A2 * D_beta
            
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * Delta_pos - Positions[i, :])
            X3 = Delta_pos - A3 * D_delta

            Candidate_GWO = (X1 + X2 + X3) / 3
            Candidate_GWO = np.clip(Candidate_GWO, lb, ub)

            if variant == 'igwo':
                # IGWO: DLH Strategy
                idxs = [k for k in range(SearchAgents_no) if k != i]
                n_idx = np.random.choice(idxs)
                r_idx = np.random.choice(idxs)
                while r_idx == n_idx: r_idx = np.random.choice(idxs)

                Candidate_DLH = Positions[i, :] + np.random.rand(dim) * (Positions[n_idx, :] - Positions[r_idx, :])
                Candidate_DLH = np.clip(Candidate_DLH, lb, ub)

                # IGWO: Greedy Selection
                fit_GWO = fobj(Candidate_GWO)
                fit_DLH = fobj(Candidate_DLH)

                if fit_GWO < fit_DLH:
                    Positions[i, :] = Candidate_GWO
                else:
                    Positions[i, :] = Candidate_DLH
            else:
                # GWO Gốc
                Positions[i, :] = Candidate_GWO

        Convergence_curve[t] = Alpha_score
        
    return Alpha_score, Alpha_pos, Convergence_curve

if __name__ == "__main__":
    # Hàm Rastrigin (Nhiều cực trị địa phương để test khả năng thoát bẫy)
    def rastrigin_func(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    SearchAgents_no = 30
    Max_iter = 200
    dim_cont = 30
    lb_cont = -5.12
    ub_cont = 5.12
    
    variants_to_run = ['original', 'igwo']
    results = {}

    print("Running comparison...")
    
    for variant in variants_to_run:
        score, pos, curve = GWO(rastrigin_func, lb_cont, ub_cont, dim_cont, SearchAgents_no, Max_iter, variant=variant)
        results[variant] = {'score': score, 'curve': curve}
        print(f" -> {variant.upper()}: Best Fitness = {score:.6e}")

    plt.figure(figsize=(10, 6))
    for variant, data in results.items():
        plt.plot(data['curve'], label=variant.upper(), lw=2)
    
    plt.title('Comparison: GWO vs IGWO (Rastrigin Function)')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness (Log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()