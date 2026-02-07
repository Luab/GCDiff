import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
from node2vec import Node2Vec
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: Combine all NetworkX graphs into a union graph
# ============================================================================

def create_union_graph(graphs_list):
    """Combine all NetworkX graphs into a single union graph"""
    union_graph = nx.compose_all(graphs_list)
    print(f"Union graph created with {union_graph.number_of_nodes()} nodes and {union_graph.number_of_edges()} edges")
    return union_graph

# ============================================================================
# PART 2: Train Node2Vec on the Union Graph
# ============================================================================

def train_node2vec(union_graph, dimensions=128, walk_length=30, num_walks=200, workers=16):
    """Train Node2Vec on the union graph"""
    print(f"Training Node2Vec with dimensions={dimensions}...")
    node2vec = Node2Vec(
        union_graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=1,  # Return parameter
        q=1   # In-out parameter
    )
    model = node2vec.fit(window=10, min_count=1, epochs=10)

    # Extract node embeddings
    node_embeddings = {str(node): model.wv[str(node)] for node in union_graph.nodes()}
    print(f"Node2Vec training complete. Embeddings for {len(node_embeddings)} nodes extracted.")
    return node_embeddings, model

# ============================================================================
# PART 3: Simple Edge Type Embedding (if Edge2Vec not available)
# ============================================================================

def create_edge_embeddings_from_nodes(union_graph, node_embeddings, method='concatenate'):
    """
    Create edge embeddings by combining node embeddings.
    Methods: 'concatenate', 'average', 'hadamard'
    """
    edge_embeddings = {}
    edge_embedding_dim = None

    for u, v, data in union_graph.edges(data=True):
        u_emb = torch.tensor(node_embeddings[str(u)], dtype=torch.float)
        v_emb = torch.tensor(node_embeddings[str(v)], dtype=torch.float)

        if method == 'concatenate':
            edge_emb = torch.cat([u_emb, v_emb])
        elif method == 'average':
            edge_emb = (u_emb + v_emb) / 2.0
        elif method == 'hadamard':
            edge_emb = u_emb * v_emb
        else:
            raise ValueError(f"Unknown method: {method}")

        edge_embeddings[(u, v)] = edge_emb.numpy()
        edge_embedding_dim = edge_emb.shape[0]

    print(f"Edge embeddings created ({len(edge_embeddings)} edges) using {method} method. Dimension: {edge_embedding_dim}")
    return edge_embeddings, edge_embedding_dim

# ============================================================================
# PART 4: Extract embedding for a specific graph
# ============================================================================

def encode_specific_graph_with_node2vec_edge2vec(
    G_specific, 
    node_embeddings, 
    edge_embeddings,
    edge_embedding_method='concatenate'
):
    """
    Encode a specific NetworkX graph using Node2Vec and Edge2Vec embeddings.
    Returns a PyG Data object ready for GNN training.
    """
    node_ids = list(G_specific.nodes())

    # Node features
    node_features = []
    for node in node_ids:
        node_vec = node_embeddings.get(str(node))
        if node_vec is None:
            raise ValueError(f"Node {node} not found in global node embeddings!")
        node_features.append(node_vec)

    x = torch.tensor(np.array(node_features), dtype=torch.float)

    # Edge features and edge_index
    edge_index_list = []
    edge_attr_list = []
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}

    for u, v in G_specific.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        edge_index_list.append([u_idx, v_idx])

        edge_key = (u, v) if (u, v) in edge_embeddings else (v, u)
        edge_feat = edge_embeddings.get(edge_key)

        if edge_feat is None:
            # Create edge embedding on the fly if not found
            u_emb = torch.tensor(node_embeddings[str(u)], dtype=torch.float)
            v_emb = torch.tensor(node_embeddings[str(v)], dtype=torch.float)
            if edge_embedding_method == 'concatenate':
                edge_feat = torch.cat([u_emb, v_emb]).numpy()
            elif edge_embedding_method == 'average':
                edge_feat = ((u_emb + v_emb) / 2.0).numpy()
            elif edge_embedding_method == 'hadamard':
                edge_feat = (u_emb * v_emb).numpy()
            edge_embeddings[(u, v)] = edge_feat

        edge_attr_list.append(edge_feat)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, x.shape[1] * 2), dtype=torch.float)  # Default to concatenate dim

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"Graph encoded: {len(node_ids)} nodes, {len(edge_index_list)} edges")
    return data

# ============================================================================
# PART 5: GNN Model for Node and Edge Embeddings
# ============================================================================

class GraphEmbeddingGNN(nn.Module):
    """
    A GNN that learns node and edge embeddings jointly.
    Can process a single graph or a batch of graphs.
    """

    def __init__(self, input_node_dim, edge_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()

        # Node layers
        self.node_convs = nn.ModuleList()
        in_channels = input_node_dim
        for i in range(num_layers):
            self.node_convs.append(GCNConv(in_channels, hidden_dim))
            in_channels = hidden_dim

        # Edge embedding layer (learns to refine edge features)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final node refinement layer
        self.node_refiner = nn.Linear(hidden_dim + hidden_dim, output_dim)  # Combine node + edge info

        self.output_dim = output_dim

    def forward(self, data):
        """
        Forward pass: encode both nodes and edges
        Input: PyG Data object (x: node features, edge_index, edge_attr)
        Output: node embeddings, edge embeddings, graph embedding
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # GCN layers for node embeddings
        node_emb = x
        for i, conv in enumerate(self.node_convs):
            node_emb = conv(node_emb, edge_index)
            if i < len(self.node_convs) - 1:
                node_emb = torch.relu(node_emb)

        # Encode edge features
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = torch.zeros((edge_index.shape[1], self.output_dim), device=x.device)
        '''
        # Refine node embeddings using edge information
        if edge_index.shape[1] > 0:
            # Aggregate edge embeddings to nodes
            edge_to_node_agg = torch.zeros((x.shape[0], edge_emb.shape[1]), device=x.device)
            edge_to_node_agg.scatter_add_(0, edge_index[0], edge_emb)
            edge_to_node_agg.scatter_add_(0, edge_index[1], edge_emb)
            edge_to_node_agg = edge_to_node_agg / (torch.clamp(torch.bincount(edge_index.flatten()), min=1).unsqueeze(1))

            node_emb_refined = self.node_refiner(torch.cat([node_emb, edge_to_node_agg], dim=1))
        else:
            # If no edges, use node embedding only
            node_emb_refined = self.node_refiner(torch.cat([node_emb, torch.zeros_like(node_emb)], dim=1))
        
        # Graph-level embedding (mean pooling)
        if hasattr(data, 'batch'):
            graph_emb = global_mean_pool(node_emb_refined, data.batch)
        else:
            graph_emb = node_emb_refined.mean(dim=0, keepdim=True)
        '''
        return {
            'node_embeddings': node_emb,#node_emb_refined,
            'edge_embeddings': edge_emb,
            #'graph_embedding': graph_emb
        }

# ============================================================================
# PART 6: Training GNN on a Single Graph
# ============================================================================

def train_gnn_on_single_graph(data, hidden_dim=128, output_dim=128, learning_rate=0.01, epochs=50):
    """
    Train a GNN model on a single graph to learn node and edge embeddings.
    """
    node_input_dim = data.x.shape[1]
    edge_input_dim = data.edge_attr.shape[1] if data.edge_attr is not None and data.edge_attr.shape[0] > 0 else 1

    model = GraphEmbeddingGNN(
        input_node_dim=node_input_dim,
        edge_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training GNN on single graph for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(data)
        node_emb = output['node_embeddings']

        # Simple reconstruction loss: try to reconstruct edge connections
        if data.edge_index.shape[1] > 0:
            # For each edge, compute similarity between endpoint node embeddings
            edge_scores = torch.sum(node_emb[data.edge_index[0]] * node_emb[data.edge_index[1]], dim=1)
            target_scores = torch.ones_like(edge_scores)  # Target: edges should be similar
            loss = torch.nn.functional.mse_loss(edge_scores, target_scores)
        else:
            loss = torch.tensor(0.0)

        if loss > 0:
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    print("GNN training complete!")
    return model

# ============================================================================
# PART 7: Extract embeddings using trained GNN
# ============================================================================

def get_graph_embeddings_from_gnn(model, data):
    """
    Use trained GNN to extract node and edge embeddings for a graph.
    """
    model.eval()
    with torch.no_grad():
        output = model(data)

    return {
        'node_embeddings': output['node_embeddings'].numpy(),
        'edge_embeddings': output['edge_embeddings'].numpy(),
        #'graph_embedding': output['graph_embedding'].numpy()
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create some example clinical graphs
    '''
    G1 = nx.Graph()
    G1.add_node("Patient_001")
    G1.add_node("Diagnosis_Flu")
    G1.add_node("Symptom_Fever")
    G1.add_edge("Patient_001", "Diagnosis_Flu", type="has_diagnosis")
    G1.add_edge("Diagnosis_Flu", "Symptom_Fever", type="causes")

    G2 = nx.Graph()
    G2.add_node("Patient_002")
    G2.add_node("Diagnosis_Flu")
    G2.add_node("Medication_Tamiflu")
    G2.add_edge("Patient_002", "Diagnosis_Flu", type="has_diagnosis")
    G2.add_edge("Diagnosis_Flu", "Medication_Tamiflu", type="treated_with")

    G3 = nx.Graph()
    G3.add_node("Patient_003")
    G3.add_node("Diagnosis_COVID")
    G3.add_node("Symptom_Cough")
    G3.add_edge("Patient_003", "Diagnosis_COVID", type="has_diagnosis")
    G3.add_edge("Diagnosis_COVID", "Symptom_Cough", type="causes")
    '''
    df = pd.read_pickle(open("reports_processed_graphs.pkl", "rb"))
    graphs_list = df

    print("=" * 80)
    print("APPROACH 1: NODE2VEC + EDGE2VEC (Pre-trained embeddings)")
    print("=" * 80)

    # Step 1: Create union graph
    union_graph = create_union_graph(graphs_list)

    # Step 2: Train Node2Vec
    node_embeddings, n2v_model = train_node2vec(union_graph, dimensions=768)

    # Step 3: Create edge embeddings
    edge_embeddings, edge_dim = create_edge_embeddings_from_nodes(
        union_graph, 
        node_embeddings, 
        method='average'
    )

    # Step 4: Encode a specific graph
    print("\nEncoding Graph 1 with Node2Vec + Edge embeddings...")
    data_g1 = encode_specific_graph_with_node2vec_edge2vec(
        df[0], 
        node_embeddings, 
        edge_embeddings,
        edge_embedding_method='average'
    )
    print(f"  Node features shape: {data_g1.x.shape}")
    print(f"  Edge features shape: {data_g1.edge_attr.shape}")
    print(f"  Edge index shape: {data_g1.edge_index.shape}")

    print(f" Embedding: {torch.cat([data_g1.x, data_g1.edge_attr], dim=0).mean(dim=0, keepdim=True).shape} ")

    '''
    print("\n" + "=" * 80)
    print("APPROACH 2: GNN-BASED EMBEDDING (Learned embeddings)")
    print("=" * 80)
    
    # Step 1: Train GNN on Graph 1
    print("\nTraining GNN on Graph 1...")
    gnn_model = train_gnn_on_single_graph(
        data_g1,
        hidden_dim=768,
        output_dim=768,
        learning_rate=0.01,
        epochs=50
    )

    # Step 2: Extract embeddings using trained GNN
    print("\nExtracting embeddings from trained GNN...")
    gnn_embeddings_g1 = get_graph_embeddings_from_gnn(gnn_model, data_g1)

    print(f"  Node embeddings shape: {gnn_embeddings_g1['node_embeddings'].shape}")
    print(f"  Edge embeddings shape: {gnn_embeddings_g1['edge_embeddings'].shape}")
    #print(f"  Graph embedding shape: {gnn_embeddings_g1['graph_embedding'].shape}")
    
    # Step 3: Get embeddings for Graph 2 using the same GNN
    print("\nEncoding Graph 2 with Node2Vec + Edge embeddings...")
    data_g2 = encode_specific_graph_with_node2vec_edge2vec(
        df[1], 
        node_embeddings, 
        edge_embeddings,
        edge_embedding_method='concatenate'
    )

    print("\nExtracting embeddings from trained GNN for Graph 2...")
    gnn_embeddings_g2 = get_graph_embeddings_from_gnn(gnn_model, data_g2)

    print(f"  Node embeddings shape: {gnn_embeddings_g2['node_embeddings'].shape}")
    print(f"  Edge embeddings shape: {gnn_embeddings_g2['edge_embeddings'].shape}")
    print(f"  Graph embedding shape: {gnn_embeddings_g2['graph_embedding'].shape}")

    print("\n" + "=" * 80)
    print("SUCCESS! Both approaches are ready for use.")
    print("=" * 80)
    print("""
SUMMARY:
--------
1. APPROACH 1 (Node2Vec + Edge2Vec):
   - Use pre-trained node embeddings for any new graph
   - Embeddings are fixed (not trained)
   - Fast inference

2. APPROACH 2 (GNN-based):
   - Train once, then use the model for any graph
   - Learn task-specific embeddings
   - More flexible and adaptive

You can combine both approaches or use one based on your needs!
    """)
    '''