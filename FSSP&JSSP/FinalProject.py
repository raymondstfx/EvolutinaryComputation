import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Random seed
random.seed(42)
np.random.seed(42)

# Load FSSP data
fssp_data_loaded = pd.read_excel('fssp_data.xlsx')
fssp_jobs = fssp_data_loaded['Job ID'].unique()
fssp_machines = sorted(fssp_data_loaded['Machine'].unique())

# Build job->operations mapping
fssp_job_operations = {}
for j in fssp_jobs:
    ops = fssp_data_loaded[fssp_data_loaded['Job ID'] == j].sort_values('Operation ID')
    fssp_job_operations[j] = list(zip(ops['Operation ID'], ops['Machine'], ops['Processing Time']))

def fssp_schedule(jobs_order, job_operations, machines):
    num_machines = len(machines)
    c_time = np.zeros((num_machines, len(jobs_order)))
    for j_idx, job in enumerate(jobs_order):
        ops = job_operations[job]
        for op_idx, (op_id, m, p_time) in enumerate(ops):
            m_idx = machines.index(m)
            if j_idx == 0 and m_idx == 0:
                c_time[m_idx, j_idx] = p_time
            elif j_idx == 0:
                c_time[m_idx, j_idx] = c_time[m_idx-1, j_idx] + p_time
            elif m_idx == 0:
                c_time[m_idx, j_idx] = c_time[m_idx, j_idx-1] + p_time
            else:
                start = max(c_time[m_idx, j_idx-1], c_time[m_idx-1, j_idx])
                c_time[m_idx, j_idx] = start + p_time
    makespan = c_time[-1, -1]
    return makespan

def fssp_init_population(pop_size=30):
    population = []
    base_jobs = list(fssp_jobs)
    for _ in range(pop_size):
        perm = base_jobs[:]
        random.shuffle(perm)
        population.append(perm)
    return population

def fssp_fitness(ind):
    return fssp_schedule(ind, fssp_job_operations, fssp_machines)

def fssp_selection(pop, scores, k=5):
    idx = np.argsort(scores)[:k]
    return [pop[i] for i in idx]

def fssp_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    for x in p2:
        if x not in child:
            for i in range(size):
                if child[i] is None:
                    child[i] = x
                    break
    return child

def fssp_mutation(ind, rate=0.1):
    if random.random() < rate:
        a, b = random.sample(range(len(ind)), 2)
        ind[a], ind[b] = ind[b], ind[a]
    return ind

# Genetic Algorithm for FSSP
fssp_pop = fssp_init_population()
fssp_generations = 30
fssp_best_scores = []
for g in range(fssp_generations):
    scores = [fssp_fitness(ind) for ind in fssp_pop]
    best = min(scores)
    fssp_best_scores.append(best) # Fitness Tracking
    selected = fssp_selection(fssp_pop, scores)
    children = []
    while len(children) < len(fssp_pop):
        p1, p2 = random.sample(selected, 2)
        c = fssp_crossover(p1, p2)
        c = fssp_mutation(c)
        children.append(c)
    fssp_pop = children

fssp_scores_final = [fssp_fitness(ind) for ind in fssp_pop]
fssp_best_ind = fssp_pop[np.argmin(fssp_scores_final)]
fssp_best_makespan = min(fssp_scores_final)

print("FSSP optimal sequence:", fssp_best_ind)
print("FSSP optimal makespan:", fssp_best_makespan)

# Plot Gantt chart for FSSP (fixed machine sequence)
def fssp_gantt(ind):
    # Calculate job intervals for machines
    order = ind
    num_machines = len(fssp_machines)
    c_time = np.zeros((num_machines, len(order)))
    start_times = {}
    for j_idx, job in enumerate(order):
        ops = fssp_job_operations[job]
        for op_idx, (op_id, m, p_time) in enumerate(ops):
            m_idx = fssp_machines.index(m)
            if j_idx == 0 and m_idx == 0:
                start = 0
                end = p_time
            elif j_idx == 0:
                start = c_time[m_idx-1, j_idx]
                end = start + p_time
            elif m_idx == 0:
                start = c_time[m_idx, j_idx-1]
                end = start + p_time
            else:
                start = max(c_time[m_idx, j_idx-1], c_time[m_idx-1, j_idx])
                end = start + p_time
            c_time[m_idx, j_idx] = end
            start_times[(job, op_id, m)] = (start, end)
    return start_times

fssp_intervals = fssp_gantt(fssp_best_ind)

plt.figure(figsize=(12, 6))
colors = sns.color_palette("hsv", len(fssp_jobs))
job_color_map = {j: colors[i] for i, j in enumerate(fssp_jobs)}
for (job, op, m), (st, ed) in fssp_intervals.items():
    plt.barh(m, ed-st, left=st, color=job_color_map[job])
    plt.text((st+ed)/2, m, f"J{job}O{op}", va='center', ha='center', color='white')
plt.xlabel("Time")
plt.ylabel("Machine")
plt.title("FSSP Optimal Solution Gantt Chart")
plt.yticks(fssp_machines, [f"M{m}" for m in fssp_machines])
plt.grid(True, axis='x')
plt.show()


# JSSP


# Load JSSP data
jssp_data_loaded = pd.read_excel('jssp_data.xlsx')
jssp_jobs = jssp_data_loaded['Job ID'].unique()
jssp_machines = sorted(jssp_data_loaded['Machine'].unique())

# Build job->operations mapping (operations must be sorted by Operation ID)
jssp_job_operations = {}
for j in jssp_jobs:
    ops = jssp_data_loaded[jssp_data_loaded['Job ID'] == j].sort_values('Operation ID')
    jssp_job_operations[j] = list(zip(ops['Operation ID'], ops['Machine'], ops['Processing Time']))

def jssp_calculate_metrics(ind):
    # Individual: {(job, op_id): start_time}
    # Calculate makespan and machine utilization
    machine_ops = {m: [] for m in jssp_machines}
    for (job, op_id), start in ind.items():
        # Find corresponding operation data
        op_data = [o for o in jssp_job_operations[job] if o[0] == op_id][0]
        m = op_data[1]
        p = op_data[2]
        machine_ops[m].append((start, start + p))
    makespan = 0
    total_proc = 0
    for m_ops in machine_ops.values():
        if m_ops:
            end_times = [op[1] for op in m_ops]
            makespan = max(makespan, max(end_times))
            total_proc += sum(end - start for start, end in m_ops)
    avg_util = total_proc / (makespan * len(jssp_machines)) if makespan > 0 else 0
    return makespan, avg_util

def jssp_fitness(ind):
    mksp, util = jssp_calculate_metrics(ind)
    return mksp, (1 - util)  # Minimize both mksp and (1 - util)

def non_dominated_sort(pop_objs):
    S = [[] for _ in pop_objs]
    n = [0] * len(pop_objs)
    front = [[]]
    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            # p dominates q
            if (pop_objs[p][0] <= pop_objs[q][0] and pop_objs[p][1] <= pop_objs[q][1]) and pop_objs[p] != pop_objs[q]:
                S[p].append(q)
            elif (pop_objs[q][0] <= pop_objs[p][0] and pop_objs[q][1] <= pop_objs[p][1]) and pop_objs[q] != pop_objs[p]:
                n[p] += 1
        if n[p] == 0:
            front[0].append(p)
    i = 0
    while front[i]:
        temp = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    temp.append(q)
        i += 1
        front.append(temp)
    front.pop()
    return front

def jssp_init_population(pop_size=40):
    # JSSP individual: Ensure operation order, operations must follow Operation ID
    population = []
    for _ in range(pop_size):
        ind = {}
        for job in jssp_jobs:
            ops = jssp_job_operations[job]
            current_start = 0
            for (op_id, m, p_time) in ops:
                # Ensure operation order: the next operation starts after the previous one finishes
                start_time = current_start + random.randint(0, 10)
                ind[(job, op_id)] = start_time
                current_start = start_time + p_time
        population.append(ind)
    return population

def jssp_crossover(p1, p2):
    # Uniform crossover
    child = {}
    for k in p1.keys():
        if random.random() > 0.5:
            child[k] = p1[k]
        else:
            child[k] = p2[k]
    return child

def jssp_mutation(ind, rate=0.1):
    # Randomly adjust start times
    for k in ind.keys():
        if random.random() < rate:
            ind[k] += random.randint(-5, 5)
            if ind[k] < 0:
                ind[k] = 0
    return ind

jssp_pop = jssp_init_population()
jssp_generations = 30
jssp_best_makespan_evo = []

for gen in range(jssp_generations):
    objs = [jssp_fitness(ind) for ind in jssp_pop]
    fronts = non_dominated_sort(objs) # Pareto Front Optimization
    best_front = fronts[0]
    # Record best makespan
    best_mksp_in_front = min(objs[i][0] for i in best_front)
    jssp_best_makespan_evo.append(best_mksp_in_front)
    # Selection
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) > len(jssp_pop):
            needed = len(jssp_pop) - len(new_pop)
            chosen = random.sample(front, needed)
            for c in chosen:
                new_pop.append(jssp_pop[c])
            break
        else:
            for c in front:
                new_pop.append(jssp_pop[c])

    while len(new_pop) < len(jssp_pop):
        p1, p2 = random.sample(new_pop, 2)
        c = jssp_crossover(p1, p2)
        c = jssp_mutation(c)
        new_pop.append(c)
    jssp_pop = new_pop

final_objs = [jssp_fitness(ind) for ind in jssp_pop]
final_fronts = non_dominated_sort(final_objs)
best_front = final_fronts[0]
best_sol_idx = best_front[0]
best_individual_jssp = jssp_pop[best_sol_idx]
best_mksp, best_loss_util = jssp_fitness(best_individual_jssp)
best_util = 1 - best_loss_util
print("Number of Pareto front solutions in JSSP:", len(best_front))
print("One Pareto front solution's makespan in JSSP:", best_mksp)
print("One Pareto front solution's average utilization:", best_util)

def update(frame):
    ax.clear()
    # Generate a Gantt chart showing the operations step by step
    machine_ops = {m: [] for m in jssp_machines}
    for (job, op_id), start in list(best_individual_jssp.items())[:frame + 1]:
        op_data = [o for o in jssp_job_operations[job] if o[0] == op_id][0]
        m = op_data[1]
        p = op_data[2]
        machine_ops[m].append((start, start + p, job, op_id))

    colors = sns.color_palette("hsv", len(jssp_jobs))
    for idx, job in enumerate(jssp_jobs):
        for m in jssp_machines:
            for (st, ed, jb, op_id) in machine_ops[m]:
                if jb == job:
                    rect = patches.Rectangle((st, m - 0.4), ed - st, 0.8, facecolor=colors[idx], edgecolor='black')
                    ax.add_patch(rect)
                    ax.text((st + ed) / 2, m, f"J{jb}O{op_id}", ha='center', va='center', color='white', fontsize=8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('JSSP Optimal Solution Gantt Chart (Pareto Front)')
    ax.set_xlim(0, max(best_individual_jssp.values()) + 20)
    ax.set_ylim(min(jssp_machines) - 0.5, max(jssp_machines) + 0.5)
    ax.set_yticks(jssp_machines)
    ax.set_yticklabels([f"M{mm}" for mm in jssp_machines])
    ax.grid(True)

# Gantt chart animation for JSSP
fig, ax = plt.subplots(figsize=(12, 8))
total_operations = len(best_individual_jssp)  # Get the number of all operations

ani = FuncAnimation(fig, update, frames=total_operations, repeat=False)

plt.show()

# Evolution of makespan during iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, jssp_generations + 1), jssp_best_makespan_evo, marker='o')
plt.title("Evolution of Pareto Front Best Makespan in JSSP")
plt.xlabel("Generation")
plt.ylabel("Best Makespan")
plt.grid(True)
plt.show()

# Pareto front distribution
final_front_objs = [final_objs[i] for i in best_front]
front_mksps = [o[0] for o in final_front_objs]
front_loss_utils = [o[1] for o in final_front_objs]

plt.figure(figsize=(10, 6))
plt.scatter([o[0] for o in final_objs], [o[1] for o in final_objs], color='grey', alpha=0.5, label='All Solutions')
plt.scatter(front_mksps, front_loss_utils, color='red', label='Pareto Front')
plt.title("Distribution of JSSP Solutions in Objective Space (makespan vs 1 - utilization)")
plt.xlabel("Makespan")
plt.ylabel("1 - Utilization")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Fitness Predictor (using JSSP makespan as the target)
# -----------------------------
print("Training Fitness Predictor...")

# Convert individuals to features
def individual_to_features(ind, jobs, jssp_job_operations):
    ops_sorted = []
    for job in sorted(jobs):
        ops_for_job = [op for op in jssp_job_operations[job]]
        for (op_id, m, p_time) in ops_for_job:
            ops_sorted.append((job, op_id))
    features = [ind[(j, oid)] for (j, oid) in ops_sorted]
    return features

X = []
y = []
for ind in jssp_pop:
    mksp, util = jssp_calculate_metrics(ind)
    feat = individual_to_features(ind, jssp_jobs, jssp_job_operations)
    X.append(feat)
    y.append(mksp)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Fitness Predictor training completed.")

predictions = model.predict(X_test)
print("Fitness Predictor predicted vs actual makespan (first 10 samples):")
for true_val, pred_val in list(zip(y_test, predictions))[:10]:
    print(f"Actual makespan: {true_val:.2f}, Predicted makespan: {pred_val:.2f}")

mae = np.mean(np.abs(np.array(y_test) - predictions))
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Visualization of prediction comparison
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title("Fitness Predictor Prediction vs Actual Makespan")
plt.xlabel("Actual Makespan")
plt.ylabel("Predicted Makespan")
plt.grid(True)
plt.show()

print("All tasks successfully completed.")
