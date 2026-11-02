import numpy as np
import random
import matplotlib.pyplot as plt

# 1. BENCHMARK FUNCTIONS (10D) AND THEIR BOUNDS
def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

BENCHMARKS = {
    'Sphere': {'func': sphere_function, 'bounds': (-5.12, 5.12)},
    'Rosenbrock': {'func': rosenbrock_function, 'bounds': (-2.048, 2.048)},
    'Rastrigin': {'func': rastrigin_function, 'bounds': (-5.12, 5.12)}
}

# 2. PARTICLE SWARM OPTIMIZATION (PSO) IMPLEMENTATION
def particle_swarm_optimization(fit_func, dim, bounds, variant='linear_star', 
                               max_evals=10000, pop_size=30):
    lower_b, upper_b = bounds
    bound_range = upper_b - lower_b
    v_max = 0.2 * bound_range
    
    # Initialize positions and initial velocity v(0) = 0
    positions = np.random.uniform(lower_b, upper_b, (pop_size, dim))
    velocities = np.zeros((pop_size, dim))
    
    # Personal bests
    pbest_positions = positions.copy()
    pbest_fitnesses = np.array([fit_func(p) for p in positions])
    
    evals = pop_size
    best_fit_so_far = np.min(pbest_fitnesses)
    history = [best_fit_so_far] * evals

    while evals < max_evals:
        # Determine inertia weight (w) and acceleration coefficients (c1, c2)
        if variant == 'linear_star':
            # Linearly decaying w from 0.9 to 0.4 according to lecture notes
            w = 0.9 - (0.9 - 0.4) * (evals / max_evals)
            c1, c2 = 2.0, 2.0
            topology = 'star'
        elif variant == 'const_star':
            w = 0.7
            c1, c2 = 1.49618, 1.49618
            topology = 'star'
        elif variant == 'const_ring':
            w = 0.7
            c1, c2 = 1.49618, 1.49618
            topology = 'ring' # Ring topology based on particle ID

        # Determine neighborhood bests (gbest_positions)
        gbest_positions = np.zeros_like(positions)
        if topology == 'star':
            global_best_idx = np.argmin(pbest_fitnesses)
            gbest_positions[:] = pbest_positions[global_best_idx]
        elif topology == 'ring':
            # Ring topology (neighborhood based on particle ID, N=1 left and right -> total 3)
            for i in range(pop_size):
                neighbors = [(i - 1) % pop_size, i, (i + 1) % pop_size]
                best_neighbor = neighbors[np.argmin(pbest_fitnesses[neighbors])]
                gbest_positions[i] = pbest_positions[best_neighbor]

        # Update velocities and positions
        r1 = np.random.uniform(0, 1, (pop_size, dim))
        r2 = np.random.uniform(0, 1, (pop_size, dim))
        
        velocities = (w * velocities + 
                      c1 * r1 * (pbest_positions - positions) + 
                      c2 * r2 * (gbest_positions - positions))
        
        # Enforce maximum velocity vmax (20% of the range)
        velocities = np.clip(velocities, -v_max, v_max)
        
        # Move particles (particles are allowed to fly out of bounds)
        positions = positions + velocities

        # Evaluate only particles that are inside the valid search space
        for i in range(pop_size):
            if evals >= max_evals:
                break
            
            is_inside = np.all(positions[i] >= lower_b) and np.all(positions[i] <= upper_b)
            
            if is_inside:
                f_val = fit_func(positions[i])
                evals += 1
                
                if f_val < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = f_val
                    pbest_positions[i] = positions[i].copy()
                    
                if f_val < best_fit_so_far:
                    best_fit_so_far = f_val

            history.append(best_fit_so_far)

    return history[:max_evals], best_fit_so_far

# 3. RUN EXPERIMENTS AND PLOT GRAPHS & TABLE IN A SINGLE WINDOW
def run_pso_experiments():
    dim = 10
    max_evals = 10000
    runs = 10
    pop_size = 30
    
    pso_variants = [
        ('Lin. w (0.9->0.4) Star', 'linear_star'),
        ('Const. w=0.7 Star', 'const_star'),
        ('Const. w=0.7 Ring (k=3)', 'const_ring')
    ]
    
    result_rows = []

    for f_idx, (f_name, f_info) in enumerate(BENCHMARKS.items()):
        print(f"---> Running PSO for {f_name} ({dim}D)...", flush=True)
        
        for v_label, v_type in pso_variants:
            all_histories = []
            final_bests = []
            
            for r in range(runs):
                np.random.seed(r * 100 + f_idx * 10)
                random.seed(r * 100 + f_idx * 10)
                
                hist, best = particle_swarm_optimization(
                    f_info['func'], dim, f_info['bounds'], variant=v_type,
                    max_evals=max_evals, pop_size=pop_size
                )
                
                all_histories.append(hist)
                final_bests.append(best)
                
            avg_hist = np.mean(all_histories, axis=0)
            best_res = float(np.min(final_bests))
            mean_res = float(np.mean(final_bests))
            std_res = float(np.std(final_bests))
            
            result_rows.append((f_name, v_label, best_res, mean_res, std_res))

    # Print results summary to console
    print("\n" + "=" * 85)
    print(f"{'Function':<12} | {'PSO Variant':<26} | {'Best Run':>12} | {'Mean':>12} | {'Std. Dev.':>12}")
    print("=" * 85)
    for f_name, v_label, best_res, mean_res, std_res in result_rows:
        print(f"{f_name:<12} | {v_label:<26} | {best_res:12.4e} | {mean_res:12.4e} | {std_res:12.4e}")
    print("=" * 85)

    # --- UNIFIED WINDOW FOR CONVERGENCE PLOTS AND SUMMARY TABLE ---
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3) # 2 rows: top for 3 convergence plots, bottom for the summary table

    # Plot 3 convergence graphs in the top row
    for f_idx, (f_name, f_info) in enumerate(BENCHMARKS.items()):
        ax = fig.add_subplot(gs[0, f_idx])
        
        for v_label, v_type in pso_variants:
            all_histories = []
            for r in range(runs):
                np.random.seed(r * 100 + f_idx * 10)
                random.seed(r * 100 + f_idx * 10)
                hist, _ = particle_swarm_optimization(
                    f_info['func'], dim, f_info['bounds'], variant=v_type,
                    max_evals=max_evals, pop_size=pop_size
                )
                all_histories.append(hist)
            avg_hist = np.mean(all_histories, axis=0)
            ax.plot(avg_hist, label=v_label, linewidth=1.5)
            
        ax.set_yscale('log')
        ax.set_title(f"PSO Convergence: {f_name} ({dim}D)")
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Fitness (Log Scale)")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)

    # Render summary table in the bottom row (spanning all columns)
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis('off')
    table_ax.set_title(f'PSO Results Summary ({runs} runs per variant)', pad=10, fontsize=12, fontweight='bold')
    
    table_data = [
        [f_name, v_label, f"{best:.4e}", f"{mean:.4e}", f"{std:.4e}"]
        for f_name, v_label, best, mean, std in result_rows
    ]
    
    table = table_ax.table(
        cellText=table_data,
        colLabels=['Function', 'PSO Variant', 'Best Run', 'Mean', 'Std. Deviation'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.35, 0.16, 0.16, 0.16]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.show()

if __name__ == "__main__":
    run_pso_experiments()