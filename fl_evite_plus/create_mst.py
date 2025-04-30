import json
import os
from collections import deque

def load_graph(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_tree(tree, path):
    with open(path, 'w') as f:
        json.dump(tree, f, indent=4)

def bfs_weighted_directed_tree(graph, start='0', max_root_degree=8, w_cost=1.0, w_depth=1.0, w_degree=5.0):
    visited = set([start])
    tree = {node: {} for node in graph}
    depth_map = {start: 0}
    queue = deque([start])
    root_degree = 0

    while queue:
        current = queue.popleft()
        current_depth = depth_map[current]

        neighbors = [
            (neighbor, float(cost))
            for neighbor, cost in graph[current].items()
            if neighbor not in visited
        ]

        scored_neighbors = []
        for neighbor, cost in neighbors:
            degree_penalty = 1.0 if current != start else (
                1000.0 if root_degree >= max_root_degree else 0.0
            )
            score = w_cost * cost + w_depth * (current_depth + 1) + w_degree * degree_penalty
            scored_neighbors.append((score, neighbor, cost))

        for _, neighbor, cost in sorted(scored_neighbors, key=lambda x: x[0]):
            if neighbor not in visited:
                # ➜ Only one direction: from current → neighbor
                tree[current][neighbor] = cost
                visited.add(neighbor)
                depth_map[neighbor] = current_depth + 1
                queue.append(neighbor)
                if current == start:
                    root_degree += 1

    # ➜ Clean tree: remove empty nodes (leaves)
    return {node: children for node, children in tree.items() if children}


def main():
    graph = load_graph("../graph.json")

    #(w_cost, w_depth, w_degree)
    CONFIGS = [
        (1, 1, 1), (1, 1, 5), (1, 1, 10), (1, 2, 5), (1, 3, 5),
        (2, 1, 5), (3, 1, 5), (1, 2, 10), (2, 2, 10), (3, 3, 10),
        (1, 1, 0), (0.5, 1, 10), (1, 0.5, 10), (1, 3, 0), (3, 1, 0),
        (0.5, 3, 5), (3, 0.5, 5), (2, 2, 2), (1, 4, 8), (4, 1, 8),
    ]

    CONFIGS = [(1,1,1)]

    os.makedirs("../generated_trees", exist_ok=True)

    for idx, (w_cost, w_depth, w_degree) in enumerate(CONFIGS, start=1):
        tree = bfs_weighted_directed_tree(
            graph,
            start='0',
            max_root_degree=8,
            w_cost=w_cost,
            w_depth=w_depth,
            w_degree=w_degree
        )
        filename = f"tree_config_{idx:02}_C{w_cost}_D{w_depth}_G{w_degree}.json"
        write_tree(tree, os.path.join("../generated_trees", filename))
        print(f"[✓] Saved: {filename}")

if __name__ == "__main__":
    main()
