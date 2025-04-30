import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def load_nodes(csv_path):
    df = pd.read_csv(csv_path)
    return {
        str(int(row['id'])): (float(row['x']), float(row['y']))
        for _, row in df.iterrows()
    }

def load_graph(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def draw_directed_graph(node_positions, graph, output_path="directed_graph.png", show_edges=True, title="Directed IoT Tree"):
    G = nx.DiGraph()

    # Add nodes
    for node_id, pos in node_positions.items():
        G.add_node(node_id, pos=pos)

    # Add directed edges
    if show_edges:
        for parent, children in graph.items():
            for child, cost in children.items():
                G.add_edge(parent, child, weight=round(float(cost), 2))

    pos = {node: (x, y) for node, (x, y) in node_positions.items()}

    plt.figure(figsize=(12, 12))

    # Draw nodes and labels
    node_colors = ['red' if node == '0' else 'skyblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=8)

    if show_edges:
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', connectionstyle="arc3,rad=0.05")
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Directed graph saved to {output_path}")

def main():
    csv_path = "../iot_nodes.csv"
    json_path = "../mst.json"  # path to your directed tree JSON
    output_image_path = "../directed_graph.png"
    show_edges = True

    node_positions = load_nodes(csv_path)
    graph = load_graph(json_path)
    draw_directed_graph(node_positions, graph, output_path=output_image_path, show_edges=show_edges)

if __name__ == "__main__":
    main()
