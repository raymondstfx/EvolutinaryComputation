# Depth-First Search (DFS) with cycle detection

def is_valid_state(state):
    # Check constraints: farmer cannot leave goat with wolf or cabbage with goat
    farmer, goat, wolf, cabbage = state
    if farmer != goat and (goat == wolf or goat == cabbage):
        return False
    return True

def get_next_states(state):
    farmer, goat, wolf, cabbage = state
    next_states = []
    # Farmer moves alone
    next_states.append(('w' if farmer == 'e' else 'e', goat, wolf, cabbage))
    # Farmer moves with goat
    if farmer == goat:
        next_states.append(('w' if farmer == 'e' else 'e', 'w' if goat == 'e' else 'e', wolf, cabbage))
    # Farmer moves with wolf
    if farmer == wolf:
        next_states.append(('w' if farmer == 'e' else 'e', goat, 'w' if wolf == 'e' else 'e', cabbage))
    # Farmer moves with cabbage
    if farmer == cabbage:
        next_states.append(('w' if farmer == 'e' else 'e', goat, wolf, 'w' if cabbage == 'e' else 'e'))

    # Filter valid states
    return [s for s in next_states if is_valid_state(s)]

def dfs(state, goal_state, visited, path):
    if state == goal_state:
        return [path]

    visited.add(state)
    solutions = []
    for next_state in get_next_states(state):
        if next_state not in visited:
            solutions.extend(dfs(next_state, goal_state, visited, path + [next_state]))
    visited.remove(state)

    return solutions

# Breadth-First Search (BFS) with cycle detection
from collections import deque

def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [initial_state])])
    visited = set()
    solutions = []

    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            solutions.append(path)
            continue

        visited.add(state)
        for next_state in get_next_states(state):
            if next_state not in visited and next_state not in [p[-1] for p in queue]:
                queue.append((next_state, path + [next_state]))

    return solutions

# Test both algorithms
if __name__ == "__main__":
    initial_state = ('e', 'e', 'e', 'e')
    goal_state = ('w', 'w', 'w', 'w')

    print("DFS Solutions:")
    dfs_solutions = dfs(initial_state, goal_state, set(), [initial_state])
    for i, solution in enumerate(dfs_solutions, 1):
        print(f"Solution {i}:")
        for step in solution:
            print(step)
        print()

    print("BFS Solutions:")
    bfs_solutions = bfs(initial_state, goal_state)
    for i, solution in enumerate(bfs_solutions, 1):
        print(f"Solution {i}:")
        for step in solution:
            print(step)
        print()
