import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def generate_moves(x, y):
    moves = [
        (x + 2, y + 1), (x + 2, y - 1),
        (x - 2, y + 1), (x - 2, y - 1),
        (x + 1, y + 2), (x + 1, y - 2),
        (x - 1, y + 2), (x - 1, y - 2)
    ]
    return [(a, b) for a, b in moves if 1 <= a <= 8 and 1 <= b <= 8]

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        return x

def create_graph_data():
    edge_index = []
    for i in range(1, 9):
        for j in range(1, 9):
            moves = generate_moves(i, j)
            for move in moves:
                edge_index.append((i * 8 + j - 9, move[0] * 8 + move[1] - 9))

    x = torch.eye(64)  # Each node represents a chessboard cell

    data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous())
    return data

def find_shortest_path_gcn(model, data, initial_cell, final_cell):
    initial_node = (initial_cell[0] - 1) * 8 + initial_cell[1] - 1
    final_node = (final_cell[0] - 1) * 8 + final_cell[1] - 1

    with torch.no_grad():
        output = model(data)

    # Convert the edge list to a NetworkX graph
    graph = nx.Graph()
    graph.add_edges_from(data.edge_index.numpy().T)

    # Find the shortest path using NetworkX
    path = nx.shortest_path(graph, source=initial_node, target=final_node)

    # Highlight the shortest path on the graph
    edge_colors = ['blue' if (u, v) in zip(path, path[1:]) else 'gray' for u, v in graph.edges()]

    return graph, path, edge_colors

def plot_graph(graph, edge_colors, path):
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, edge_color=edge_colors)

    # Add formatted path as labels on the nodes
    labels = {node: f"{node // 8 + 1}, {node % 8 + 1}" for node in path}
    nx.draw_networkx_labels(graph, pos, labels, font_color='red')

    plt.savefig("./shortest_path_gcn.png", format="png")
    plt.show()

    # Save the graph as a DOT file with formatted path labels
    dot_path = "shortest_path_gcn.dot"
    with open(dot_path, "w") as dot_file:
        dot_file.write("./shortest_path_gcn {\n")
        for node, label in labels.items():
            dot_file.write(f'    {node} [label="{label}"];\n')
        for edge in graph.edges():
            dot_file.write(f'    {edge[0]} -- {edge[1]};\n')
        dot_file.write("}")
      
if __name__ == "__main__":
    model = GCN(input_dim=64, hidden_dim=32, output_dim=64)
    data = create_graph_data()

    initial_cell = tuple(map(int, input("Enter initial cell (e.g., 1 1): ").split()))
    final_cell = tuple(map(int, input("Enter final cell (e.g., 8 8): ").split()))

    graph, path, edge_colors = find_shortest_path_gcn(model, data, initial_cell, final_cell)

    print("Shortest path using GCN:")
    print(path)
    
    # Format and print the path
    formatted_path = [(node // 8 + 1, node % 8 + 1) for node in path]
    print("Formatted path (Corresponding to Graph Nodes):")
    print(formatted_path)

    # Plot the graph with highlighted shortest path and formatted path labels
    plot_graph(graph, edge_colors, path)

    print("Oriented graph image has been saved as 'shortest_path_gcn.png'")
    print("Oriented graph has been saved as 'shortest_path_gcn.dot'")