"""
CheXpert Graph Conditioning Dataset Loader

Aligns three data sources for ControlNet training:
1. CheXpert chest X-ray images
2. Graph embeddings from medical reports
3. Text prompts (clinical impressions/findings)

Direct 1:1 mapping: CSV row index → HDF5 embedding index
"""

import os
import json
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
from torchvision import transforms


class CheXpertGraphDataset(Dataset):
    """
    Dataset that loads CheXpert images with corresponding graph embeddings.
    
    Args:
        csv_path: Path to reports_processed.csv (223,462 samples)
        embeddings_path: Path to embeddings file (HDF5 or pickle)
        image_root: Root directory for CheXpert images (/mnt/data/CheXpert/PNG)
        split: "train" or "valid" to filter by path_to_image prefix
        image_size: Target image size for resizing (default: 512 for RadEdit)
        text_field: Column to use for text prompts ("section_impression" or "section_findings")
        max_samples: Optional limit on dataset size (for testing)
        embedding_dim: "expanded" for [128, 768] or "original" for [768]
        load_graphs: If True, load original NetworkX graphs for counterfactual evaluation
        graphs_path: Path to pickle file containing graphs (used when load_graphs=True)
        labels_path: Path to JSON file with ground truth pathology labels (one JSON per line)
    """
    
    def __init__(
        self,
        csv_path: str = "/home/luab/graph/reports_processed.csv",
        embeddings_path: str = "/home/luab/graph/new_embeddings_expanded.h5",
        image_root: str = "/mnt/data/CheXpert/PNG",
        split: str = "train",
        image_size: int = 512,
        text_field: str = "section_impression",
        max_samples: Optional[int] = None,
        embedding_dim: str = "expanded",
        load_graphs: bool = False,
        graphs_path: str = "/mnt/data/diffusion_graph/reports_processed_graphs.pkl",
        labels_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Validate embedding_dim
        valid_dims = ["expanded", "original"]
        if embedding_dim not in valid_dims:
            raise ValueError(f"embedding_dim must be one of {valid_dims}, got {embedding_dim}")
        
        print(f"Loading CheXpertGraphDataset (split={split}, embedding_dim={embedding_dim})...")
        
        # Load CSV metadata
        self.df = pd.read_csv(csv_path)
        print(f"  Loaded CSV: {len(self.df)} total samples")
        
        # Filter by split (train/valid based on path_to_image prefix)
        self.df = self.df[self.df['path_to_image'].str.startswith(f"{split}/")]
        print(f"  After {split} filter: {len(self.df)} samples")
        
        # Reset index but keep original index as column for HDF5 lookup
        self.df = self.df.reset_index(drop=False)
        self.df = self.df.rename(columns={'index': 'original_idx'})
        
        # Limit samples if specified (for testing)
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]
            print(f"  Limited to {max_samples} samples for testing")
        
        # Open embeddings file (HDF5 or pickle)
        print(f"  Opening embeddings: {embeddings_path}")
        self.embedding_dim = embedding_dim
        self.embeddings_path = embeddings_path
        
        if embeddings_path.endswith('.h5') or embeddings_path.endswith('.hdf5'):
            # HDF5 file
            self.embeddings_file = h5py.File(embeddings_path, 'r')
            self.embeddings = self.embeddings_file['embeddings']
            print(f"  HDF5 shape: {self.embeddings.shape}")
            
            # Validate shape
            expected_shape = (len(self.embeddings), 128, 768) if embedding_dim == "expanded" else (len(self.embeddings), 768)
            if self.embeddings.shape != expected_shape:
                print(f"  Warning: Expected shape {expected_shape} for embedding_dim={embedding_dim}, got {self.embeddings.shape}")
        else:
            # Pickle file - load into memory (for [768] embeddings)
            import pickle
            self.embeddings = pickle.load(open(embeddings_path, 'rb'))
            print(f"  Pickle loaded: {self.embeddings.shape}")
            self.embeddings_file = None  # No file handle to close
        
        # Load graphs if requested (for counterfactual evaluation)
        self.graphs = None
        if load_graphs:
            import pickle
            print(f"  Loading graphs: {graphs_path}")
            self.graphs = pickle.load(open(graphs_path, 'rb'))
            print(f"  Loaded {len(self.graphs)} graphs")
        
        # Load labels if provided (for counterfactual direction inference)
        self.labels = None
        if labels_path:
            print(f"  Loading labels: {labels_path}")
            self.labels = self._load_labels(labels_path)
            print(f"  Loaded labels for {len(self.labels)} samples")
        
        # Store configuration
        self.image_root = image_root
        self.text_field = text_field
        self.image_size = image_size
        self.split = split
        
        # Image transforms for RadEdit compatibility
        # RadEdit expects: [3, 512, 512] normalized to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
        ])
        
        print(f"✓ Dataset ready: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_labels(self, labels_path: str) -> List[Dict]:
        """
        Load pathology labels from JSON file (one JSON object per line).
        
        Args:
            labels_path: Path to JSON file with labels
            
        Returns:
            List of label dictionaries indexed by original CSV row
        """
        labels = []
        with open(labels_path, 'r') as f:
            for line in f:
                labels.append(json.loads(line.strip()))
        return labels
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a sample with image, graph embedding, and text prompt.
        
        Returns:
            dict: {
                'image': torch.Tensor [3, H, W] in [-1, 1],
                'graph_embedding': torch.Tensor [128, 768] or [768] depending on embedding_dim,
                'text_prompt': str,
                'patient_id': str,
                'image_path': str,
                'original_idx': int (for debugging HDF5 alignment),
                'graph': nx.DiGraph (only if load_graphs=True),
                'labels': dict (only if labels_path provided) - pathology -> value
            }
        """
        row = self.df.iloc[idx]
        original_idx = int(row['original_idx'])  # Original CSV index for embedding lookup
        
        # 1. Load image
        # Note: CSV has .jpg extension but actual files are .png
        img_path = os.path.join(self.image_root, row['path_to_image'])
        img_path = img_path.replace('.jpg', '.png')  # Fix extension mismatch
        
        try:
            # Load as PIL Image, convert to RGB
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Return black image as fallback
            image_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # 2. Load graph embedding
        if isinstance(self.embeddings, np.ndarray):
            # Numpy array (from pickle)
            graph_embedding = torch.from_numpy(
                self.embeddings[original_idx].astype(np.float32)
            )
        else:
            # HDF5 dataset
            graph_embedding = torch.from_numpy(
                self.embeddings[original_idx].astype(np.float32)
            )
        
        # 3. Extract text prompt (handle NaN/empty values)
        text_prompt = row[self.text_field]
        if pd.isna(text_prompt) or text_prompt == "":
            # Fallback to other text fields
            if self.text_field == "section_impression" and 'section_findings' in row:
                text_prompt = row['section_findings']
            if pd.isna(text_prompt) or text_prompt == "":
                text_prompt = "Chest X-ray"  # Final fallback
        
        # Clean text prompt (remove excessive whitespace, newlines)
        text_prompt = ' '.join(str(text_prompt).split())
        
        result = {
            'image': image_tensor,  # [3, 512, 512] in [-1, 1]
            'graph_embedding': graph_embedding,  # [128, 768]
            'text_prompt': text_prompt,  # str
            'patient_id': row['deid_patient_id'],
            'image_path': row['path_to_image'],
            'original_idx': original_idx,
        }
        
        # Include original graph if loaded (for counterfactual evaluation)
        if self.graphs is not None:
            result['graph'] = self.graphs[original_idx]
        
        # Include labels if loaded (for counterfactual direction inference)
        if self.labels is not None:
            result['labels'] = self.labels[original_idx]
        
        return result
    
    def __del__(self):
        """Close HDF5 file handle on cleanup."""
        if hasattr(self, 'embeddings_file') and self.embeddings_file is not None:
            self.embeddings_file.close()


def get_dataloaders(
    batch_size: int = 4,
    num_workers: int = 4,
    csv_path: str = "/home/luab/graph/reports_processed.csv",
    embeddings_path: str = "/home/luab/graph/new_embeddings_expanded.h5",
    image_root: str = "/mnt/data/CheXpert/PNG",
    image_size: int = 512,
    text_field: str = "section_impression",
    max_samples: Optional[int] = None,
    embedding_dim: str = "expanded",
    load_graphs: bool = False,
    graphs_path: str = "/mnt/data/diffusion_graph/reports_processed_graphs.pkl",
    labels_path: Optional[str] = None,
):
    """
    Create train and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        csv_path: Path to reports CSV
        embeddings_path: Path to embeddings file (HDF5 or pickle)
        image_root: Root directory for images
        image_size: Target image size
        text_field: Text field to use for prompts
        max_samples: Optional limit on dataset size (for testing)
        embedding_dim: "expanded" for [128, 768] or "original" for [768]
        load_graphs: If True, load original NetworkX graphs for counterfactual evaluation
        graphs_path: Path to pickle file containing graphs (used when load_graphs=True)
        labels_path: Path to JSON file with ground truth pathology labels
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CheXpertGraphDataset(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        image_root=image_root,
        split="train",
        image_size=image_size,
        text_field=text_field,
        max_samples=max_samples,
        embedding_dim=embedding_dim,
        load_graphs=load_graphs,
        graphs_path=graphs_path,
        labels_path=labels_path,
    )
    
    val_dataset = CheXpertGraphDataset(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        image_root=image_root,
        split="valid",
        image_size=image_size,
        text_field=text_field,
        max_samples=max_samples // 10 if max_samples else None,  # 10% for val
        embedding_dim=embedding_dim,
        load_graphs=load_graphs,
        graphs_path=graphs_path,
        labels_path=labels_path,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch size in training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset loading."""
    print("=" * 80)
    print("Testing CheXpertGraphDataset")
    print("=" * 80)
    
    # Create dataset with small sample for testing
    dataset = CheXpertGraphDataset(
        split="train",
        max_samples=10,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test loading first sample
    print("\nLoading first sample...")
    sample = dataset[0]
    
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Graph embedding shape: {sample['graph_embedding'].shape}")
    print(f"  Text prompt: {sample['text_prompt'][:100]}...")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Image path: {sample['image_path']}")
    print(f"  Original index: {sample['original_idx']}")
    
    # Test batch loading
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"  Batch image shape: {batch['image'].shape}")
    print(f"  Batch graph embedding shape: {batch['graph_embedding'].shape}")
    print(f"  Batch text prompts: {len(batch['text_prompt'])} prompts")
    
    print("\n✓ Dataset test passed!")

