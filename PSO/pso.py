# -*- coding: utf-8 -*-
"""
Asn3: Particle Swarm Optimization (PSO)
Author: Xiyao Huang
Student ID: 202306631
Email: x2023fkv@stfx.ca
Date: 2024-11-27
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import random
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set default font to SimHei
plt.rcParams['axes.unicode_minus'] = False    # Solve minus sign display issue

# -------------------- Define Test Functions --------------------

def sphere(x):
    """Sphere function for single-objective optimization"""
    return np.sum(x**2)

def zdt1(x):
    """ZDT1 function for multi-objective optimization"""
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x)-1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return f1, f2

# -------------------- Define Particle Class --------------------

class Particle:
    def __init__(self, dim, bounds):
        """Initialize particle position and velocity"""
        self.position = np.array([random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)])
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')  # Single-objective optimization
        self.value = float('inf')
        self.pareto = []  # Pareto solutions for multi-objective optimization

# -------------------- Define PSO Algorithm Class --------------------

class PSO:
    def __init__(self, func, dim, bounds, num_particles=30, max_iter=100,
                 w=0.5, c1=1.5, c2=1.5, velocity_clamp=None,
                 inertia_weight_decay=False, adaptive_params=False,
                 dynamic_search_space=False, multi_objective=False):
        """
        Initialize PSO parameters
        :param func: Objective function
        :param dim: Dimensionality
        :param bounds: Search space boundaries
        :param num_particles: Number of particles
        :param max_iter: Maximum number of iterations
        :param w: Inertia weight
        :param c1: Cognitive parameter
        :param c2: Social parameter
        :param velocity_clamp: Velocity clamp
        :param inertia_weight_decay: Whether to decay inertia weight
        :param adaptive_params: Whether to adaptively adjust parameters
        :param dynamic_search_space: Whether to dynamically adjust the search space
        :param multi_objective: Whether it is a multi-objective optimization
        """
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.velocity_clamp = velocity_clamp  # Velocity clamp
        self.inertia_weight_decay = inertia_weight_decay  # Inertia weight decay
        self.adaptive_params = adaptive_params  # Adaptive parameters
        self.dynamic_search_space = dynamic_search_space  # Dynamic search space
        self.multi_objective = multi_objective  # Multi-objective flag
        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best = None  # Global best for single-objective
        self.pareto_front = []  # Pareto front for multi-objective
        self.history = []  # History for visualization

    def optimize(self):
        """Execute PSO optimization process"""
        for iter in range(self.max_iter):
            for particle in self.particles:
                if self.multi_objective:
                    particle.value = self.func(particle.position)
                    # Update personal best (Pareto dominance)
                    if not particle.pareto:
                        particle.pareto.append(particle.value)
                        particle.best_position = np.copy(particle.position)
                    else:
                        dominated = False
                        non_dominated = []
                        for val in particle.pareto:
                            if self.dominates(val, particle.value):
                                dominated = True
                                break
                            elif not self.dominates(particle.value, val):
                                non_dominated.append(val)
                        if not dominated:
                            particle.pareto.append(particle.value)
                            particle.pareto = non_dominated
                else:
                    particle.value = self.func(particle.position)
                    # Update personal best
                    if particle.value < particle.best_value:
                        particle.best_value = particle.value
                        particle.best_position = np.copy(particle.position)

            # Update global best or Pareto front
            if self.multi_objective:
                self.update_pareto_front()
            else:
                best_particle = min(self.particles, key=lambda p: p.best_value)
                if self.global_best is None or best_particle.best_value < self.global_best.value:
                    self.global_best = best_particle

            # Dynamic adjustment of inertia weight (Enhancement 1: inertia weight decay)
            if self.inertia_weight_decay:
                self.w = self.w * 0.99  # Decay inertia weight each iteration

            # Adaptive parameter adjustment (Enhancement 2: adaptive parameters)
            if self.adaptive_params:
                self.c1 = 1.5 + 0.5 * np.sin(iter / self.max_iter * np.pi)
                self.c2 = 1.5 + 0.5 * np.cos(iter / self.max_iter * np.pi)

            # Update velocity and position
            for particle in self.particles:
                if self.multi_objective:
                    # For multi-objective, skip velocity updates
                    continue
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                if self.global_best is not None:
                    social = self.c2 * r2 * (self.global_best.position - particle.position)
                else:
                    social = 0
                particle.velocity = self.w * particle.velocity + cognitive + social

                # Enhancement 3: velocity clamping
                if self.velocity_clamp is not None:
                    particle.velocity = np.clip(particle.velocity, self.velocity_clamp[0], self.velocity_clamp[1])

                particle.position += particle.velocity

                # Dynamic adjustment of search space (Enhancement 4: dynamic search space)
                if self.dynamic_search_space:
                    for d in range(self.dim):
                        if particle.position[d] < self.bounds[d][0]:
                            particle.position[d] = self.bounds[d][0]
                            particle.velocity[d] = 0
                        elif particle.position[d] > self.bounds[d][1]:
                            particle.position[d] = self.bounds[d][1]
                            particle.velocity[d] = 0

            # Record history
            if not self.multi_objective:
                self.history.append(self.global_best.value)
            else:
                self.history.append(self.pareto_front.copy())

        if self.multi_objective:
            return self.pareto_front
        else:
            return self.global_best.value

    def dominates(self, a, b):
        """Determine if solution a dominates solution b (multi-objective optimization)"""
        return all(a_i <= b_i for a_i, b_i in zip(a, b)) and any(a_i < b_i for a_i, b_i in zip(a, b))

    def update_pareto_front(self):
        """Update the current Pareto front"""
        current_front = []
        for particle in self.particles:
            for val in particle.pareto:
                dominated = False
                for other_particle in self.particles:
                    for other_val in other_particle.pareto:
                        if self.dominates(other_val, val):
                            dominated = True
                            break
                    if dominated:
                        break
                if not dominated:
                    current_front.append(val)
        # Remove duplicates
        self.pareto_front = [list(t) for t in {tuple(t) for t in current_front}]

# -------------------- Visualization Functions --------------------

def visualize_convergence(history, title="Convergence Curve"):
    """Visualize convergence curve for single-objective optimization"""
    plt.figure(figsize=(10,5))
    plt.plot(history, label='Best Fitness')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_pareto(pareto_front, title="Pareto Front"):
    """Visualize Pareto front for multi-objective optimization"""
    f1 = [pt[0] for pt in pareto_front]
    f2 = [pt[1] for pt in pareto_front]
    plt.figure(figsize=(7,7))
    plt.scatter(f1, f2, c='r', marker='o')
    plt.title(title)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid()
    plt.show()

# -------------------- Statistical Analysis Functions --------------------

def statistical_analysis(func, dim, bounds, num_particles, max_iter, velocity_clamp1, velocity_clamp2, runs=30):
    """
    Compare the performance of two PSO variants with different velocity clamps
    :param func: Objective function
    :param dim: Dimensionality
    :param bounds: Search space boundaries
    :param num_particles: Number of particles
    :param max_iter: Maximum number of iterations
    :param velocity_clamp1: Velocity clamp for the first PSO
    :param velocity_clamp2: Velocity clamp for the second PSO
    :param runs: Number of runs
    """
    results1 = []
    results2 = []
    for _ in range(runs):
        # PSO variant 1
        pso1 = PSO(func=func, dim=dim, bounds=bounds, num_particles=num_particles, max_iter=max_iter,
                   velocity_clamp=velocity_clamp1)
        best1 = pso1.optimize()
        results1.append(best1)

        # PSO variant 2
        pso2 = PSO(func=func, dim=dim, bounds=bounds, num_particles=num_particles, max_iter=max_iter,
                   velocity_clamp=velocity_clamp2)
        best2 = pso2.optimize()
        results2.append(best2)

    # Calculate mean and standard deviation
    mean1 = np.mean(results1)
    std1 = np.std(results1)
    mean2 = np.mean(results2)
    std2 = np.std(results2)
    print(f"PSO1 - Velocity Clamp: {velocity_clamp1} Mean: {mean1:.4f}, Std Dev: {std1:.4f}")
    print(f"PSO2 - Velocity Clamp: {velocity_clamp2} Mean: {mean2:.4f}, Std Dev: {std2:.4f}")

    # Perform t-test
    t_stat, t_p = ttest_ind(results1, results2)
    print(f"T-test: Statistic={t_stat:.4f}, p-value={t_p:.4f}")

    # Perform Mann-Whitney U test
    u_stat, u_p = mannwhitneyu(results1, results2)
    print(f"Mann-Whitney U test: Statistic={u_stat}, p-value={u_p:.4f}")

    # Calculate Cohen's d
    cohen_d = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
    print(f"Cohen's d: {cohen_d:.4f}")

# -------------------- Main Function --------------------

def main():
    # Single-objective optimization example
    print("Running single-objective optimization (Sphere function)...")
    dim = 30
    bounds = [(-5.12, 5.12) for _ in range(dim)]
    pso = PSO(func=sphere, dim=dim, bounds=bounds, num_particles=30, max_iter=100,
              velocity_clamp=(-1,1), inertia_weight_decay=True, adaptive_params=True,
              dynamic_search_space=True, multi_objective=False)
    best = pso.optimize()
    print(f"Best fitness for Sphere function: {best}")
    visualize_convergence(pso.history, title="Convergence Curve for Sphere Function")

    # Multi-objective optimization example
    print("\nRunning multi-objective optimization (ZDT1 function)...")
    dim_mo = 30
    bounds_mo = [(0,1) for _ in range(dim_mo)]
    pso_mo = PSO(func=zdt1, dim=dim_mo, bounds=bounds_mo, num_particles=30, max_iter=100,
                multi_objective=True)
    pareto = pso_mo.optimize()
    print(f"Number of Pareto solutions for ZDT1 function: {len(pareto)}")
    visualize_pareto(pareto, title="Pareto Front for ZDT1 Function")

    # Statistical analysis example
    print("\nPerforming statistical analysis (comparing two velocity clamps on Sphere function)...")
    statistical_analysis(func=sphere, dim=dim, bounds=bounds, num_particles=30, max_iter=100,
                        velocity_clamp1=(-1,1), velocity_clamp2=(-2,2), runs=30)

if __name__ == "__main__":
    main()
