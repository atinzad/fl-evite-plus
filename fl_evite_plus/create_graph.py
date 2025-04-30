import csv
import math
import json
from collections import defaultdict

def load_nodes_from_csv(csv_path):
    """
    Load nodes from a CSV file.
    Each row should have: id,x,y
    Returns a dictionary mapping node ids to their (x, y) coordinates.
    """
    nodes = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            node_id = int(row['id'])
            x = float(row['x'])
            y = float(row['y'])
            nodes[node_id] = (x, y)
    return nodes

def euclidean_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two coordinates.
    """
    return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

def build_graph(nodes, threshold):
    """
    Build a graph where edges connect nodes within the threshold distance.
    The edge weight is the Euclidean distance between nodes.
    Returns a dictionary representing the graph.
    """
    graph = defaultdict(dict)
    node_ids = list(nodes.keys())
    for i, id1 in enumerate(node_ids):
        for id2 in node_ids[i+1:]:
            dist = euclidean_distance(nodes[id1], nodes[id2])
            if dist <= threshold:
                graph[id1][id2] = dist
                graph[id2][id1] = dist  # Since the graph is undirected
    return dict(graph)

def write_graph_to_json(graph, output_path):
    """
    Write the graph dictionary to a JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(graph, json_file, indent=4)

def main(csv_path, threshold, output_json_path):
    """
    Main function to load nodes, build the graph, and write it to a JSON file.
    """
    nodes = load_nodes_from_csv(csv_path)
    graph = build_graph(nodes, threshold)
    write_graph_to_json(graph, output_json_path)
    print(f"Graph has been written to {output_json_path}")

# Example usage:
if __name__ == "__main__":
    csv_file_path = '../iot_nodes.csv'      
    discovery_threshold = 10.0           # Set your desired threshold
    output_json_file = '../graph.json'      # Output JSON file path
    main(csv_file_path, discovery_threshold, output_json_file)
