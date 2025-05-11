
import json
import os
from collections import deque

def load_graph(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_tree(tree, path):
    with open(path, 'w') as f:
        json.dump(tree, f, indent=4)

def generate_layered_tree(graph, levels, root='0'):
    visited = set([root])
    tree = {root: {}}
    available_nodes = set(graph.keys()) - {root}
    
    # Start from root
    current_level_nodes = [root]

    for depth, max_children in enumerate(levels):
        next_level_nodes = []

        for parent in current_level_nodes:
            if parent not in graph:
                continue

            # Get unvisited neighbors sorted by cost
            candidate_neighbors = [
                (neighbor, float(cost))
                for neighbor, cost in graph[parent].items()
                if neighbor not in visited
            ]
            candidate_neighbors.sort(key=lambda x: x[1])

            # Select up to max_children nodes
            selected = candidate_neighbors[:max_children]

            for child, cost in selected:
                # Initialize tree structure
                if parent not in tree:
                    tree[parent] = {}
                tree[parent][child] = cost
                visited.add(child)
                next_level_nodes.append(child)

        current_level_nodes = next_level_nodes

        if not current_level_nodes:
            break  # No more nodes to expand

    return tree

def main():
    graph = load_graph("../graph.json")

    #(w_cost, w_depth, w_degree)
    

    CONFIGS = [[40],[20,1],[10,3],[10,2,1], [10,1,2]]

    os.makedirs("../generated_trees", exist_ok=True)

    for idx, levels in enumerate(CONFIGS, start=1):
        tree = generate_layered_tree(graph, levels, root='0')
        filename = f"tree_config_{idx:02}_{"_".join([str(i) for i in levels])}.json"
        write_tree(tree, os.path.join("../generated_trees", filename))
        print(f"[✓] Saved: {filename}")

if __name__ == "__main__":
    main()
