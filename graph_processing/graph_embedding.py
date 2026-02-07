import networkx as nx
import numpy as np
from karateclub import Graph2Vec
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib 
from pathlib import Path
import h5py

class SimpleGraph2VecEmbedder:
    """
    Simple wrapper for graph2vec embedding using the karateclub library.
    Handles automatic node relabeling for compatibility.
    """
    
    def __init__(self, dimensions=128, wl_iterations=2, epochs=10):
        """
        Initialize Graph2Vec model.
        
        Args:
            dimensions: Embedding dimension size
            wl_iterations: Number of Weisfeiler-Lehman iterations
            epochs: Number of training epochs
        """
        self.model = Graph2Vec(
            dimensions=dimensions,
            wl_iterations=wl_iterations,
            epochs=epochs
        )
        self.is_trained = False
        
    def _prepare_graphs(self, graphs):
        """
        Prepare graphs by relabeling nodes to consecutive integers.
        
        Args:
            graphs: List of NetworkX graphs
            
        Returns:
            List of relabeled graphs
        """
        prepared_graphs = []
        for G in graphs:
            # Relabel nodes to consecutive integers starting from 0
            G_relabeled = nx.convert_node_labels_to_integers(
                G, 
                first_label=0, 
                ordering='default'
            )
            prepared_graphs.append(G_relabeled)
        
        return prepared_graphs
        
    def fit(self, graphs):
        """
        Train the model on a list of NetworkX graphs.
        
        Args:
            graphs: List of NetworkX graphs
            
        Returns:
            self
        """
        if not isinstance(graphs, list):
            graphs = [graphs]
        
        # Prepare graphs with proper node indexing
        prepared_graphs = self._prepare_graphs(graphs)
        
        print(f"Training Graph2Vec on {len(prepared_graphs)} graphs...")
        self.model.fit(prepared_graphs)
        self.is_trained = True
        print("Training completed!")
        
        return self
    
    def encode(self, graphs=None):
        """
        Encode graphs into embeddings.
        
        Args:
            graphs: List of NetworkX graphs. If None, returns embeddings 
                   of the training graphs.
                   
        Returns:
            numpy array of shape (num_graphs, dimensions)
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before encoding")
        
        if graphs is None:
            # Return training embeddings
            return self.model.get_embedding()
        
        if not isinstance(graphs, list):
            graphs = [graphs]
        
        # Prepare graphs with proper node indexing
        prepared_graphs = self._prepare_graphs(graphs)
        
        # Infer embeddings for new graphs
        embeddings = self.model.infer(prepared_graphs)
        
        return embeddings
    
    def get_training_embeddings(self):
        """
        Get embeddings of the training graphs.
        
        Returns:
            numpy array of shape (num_training_graphs, dimensions)
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted first")
        
        return self.model.get_embedding()

class EmbeddingExpander(nn.Module):
    """
    Simpler version that directly transforms [768] -> [128, 768]
    """
    def __init__(self, input_dim=768, output_length=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_length * input_dim)
        self.output_length = output_length
        self.input_dim = input_dim
        
    def forward(self, x):
        # Convert NumPy to PyTorch if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Flatten if needed: [1, 768] -> [768]
        if x.dim() > 1:
            x = x.squeeze()
        
        # Transform: [768] -> [128*768]
        out = self.linear(x)
        
        # Reshape: [128*768] -> [128, 768]
        out = out.view(self.output_length, self.input_dim)
        
        return out

def save_with_hdf5(input_file, output_file, batch_size=100):
    """
    Process and save embeddings using HDF5 for memory efficiency
    """
    new_embeddings = pickle.load(open(input_file, "rb"))
    expander = EmbeddingExpander(input_dim=768, output_length=128)
    expander.eval()  # Set to evaluation mode
    
    num_embeddings = len(new_embeddings)
    
    # Create HDF5 file with the final shape
    with h5py.File(output_file, 'w') as f:
        # Pre-allocate dataset
        dset = f.create_dataset(
            'embeddings', 
            shape=(num_embeddings, 128, 768),
            dtype='float32',
            chunks=(1, 128, 768),  # Chunk by individual embeddings
            compression='gzip',  # Optional: compress to save disk space
            compression_opts=4
        )
        
        # Process in batches
        with torch.no_grad():  # Disable gradient computation
            for i in tqdm(range(0, num_embeddings, batch_size)):
                end_idx = min(i + batch_size, num_embeddings)
                batch = new_embeddings[i:end_idx]
                
                # Process batch
                expanded_batch = []
                for emb in batch:
                    expanded = expander(emb).detach().cpu().numpy()
                    expanded_batch.append(expanded)
                
                # Write directly to disk
                dset[i:end_idx] = np.array(expanded_batch)
                
                # Clear memory
                del expanded_batch
                
    print(f"Saved {num_embeddings} embeddings to {output_file}")


# Example usage
if __name__ == "__main__":
    print("=== Simple Graph2Vec Example ===\n")
    # Create graphs
    '''
    #Block 1 - Load graphs and training the embedder
    graphs = pickle.load(open("reports_processed_graphs.pkl", "rb"))
    
    # Initialize and train
    embedder = SimpleGraph2VecEmbedder(dimensions=768, wl_iterations=2, epochs=10)
    embedder.fit(graphs)
    pickle.dump(embedder, open("graph2vec_embeddings.pkl", "wb"))
    
    
    '''
    # Block 2 - Load trained embedder and get training embeddings
    graphs = pickle.load(open("reports_processed_graphs.pkl", "rb"))
    embedder = pickle.load(open("graph2vec_embeddings.pkl", "rb"))
   
    new_embeddings = embedder.encode(graphs)
    pickle.dump(new_embeddings, open("new_embeddings_768.pkl", "wb"))
    '''
    '''
    # Block 3 - Expand embeddings and save with HDF5
    save_with_hdf5("new_embeddings_768.pkl", "new_embeddings_expanded.h5", batch_size=100)
    