import networkx as nx
from typing import List, Set, Tuple, Dict, Optional, Union
import pickle

class MedicalGraphManager:
    """
    Manager for handling multiple NetworkX graphs containing medical entities and relations.
    Supports aggregation of nodes/edges and instance-level editing with new graph generation.
    """
    
    def __init__(self, graphs: List[nx.DiGraph]):
        """
        Initialize with a list of NetworkX graphs.
        
        Args:
            graphs: List of NetworkX DiGraph objects (supports any graph type)
        """
        self.graphs = graphs
        self._aggregate_cache = None
    
    def get_all_nodes(self, refresh: bool = False) -> Set:
        """
        Get all unique nodes across all graphs.
        
        Args:
            refresh: Force recalculation (bypasses cache)
            
        Returns:
            Set of all unique nodes
        """
        if self._aggregate_cache is not None and not refresh:
            return self._aggregate_cache['nodes']
        
        all_nodes = set()
        for graph in self.graphs:
            all_nodes.update(graph.nodes())
        
        return all_nodes
    
    def get_all_edges(self, refresh: bool = False) -> Set[Tuple]:
        """
        Get all unique edges across all graphs.
        
        Args:
            refresh: Force recalculation (bypasses cache)
            
        Returns:
            Set of all unique edges (as tuples)
        """
        if self._aggregate_cache is not None and not refresh:
            return self._aggregate_cache['edges']
        
        all_edges = set()
        for graph in self.graphs:
            all_edges.update(graph.edges())
        
        return all_edges
    
    def get_node_attributes(self, node: str) -> Dict:
        """
        Get attributes of a node from all graphs where it exists.
        
        Args:
            node: Node identifier
            
        Returns:
            Dictionary with graph index as key and attributes as value
        """
        attributes = {}
        for idx, graph in enumerate(self.graphs):
            if node in graph.nodes():
                attributes[f'graph_{idx}'] = dict(graph.nodes[node])
        
        return attributes
    
    def get_edge_attributes(self, source: str, target: str) -> Dict:
        """
        Get attributes of an edge from all graphs where it exists.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            Dictionary with graph index as key and attributes as value
        """
        attributes = {}
        for idx, graph in enumerate(self.graphs):
            if graph.has_edge(source, target):
                attributes[f'graph_{idx}'] = dict(graph.edges[source, target])
        
        return attributes
    
    def edit_instance(self, 
                     graph_index: int,
                     add_nodes: Optional[List[Tuple]] = None,
                     remove_nodes: Optional[List] = None,
                     add_edges: Optional[List[Tuple]] = None,
                     remove_edges: Optional[List[Tuple]] = None) -> nx.DiGraph:
        """
        Edit a specific graph instance by adding/removing nodes and edges.
        Returns a new modified graph without changing the original.
        
        Args:
            graph_index: Index of the graph to edit
            add_nodes: List of tuples (node_id, attributes_dict) or just node_ids
            remove_nodes: List of node IDs to remove
            add_edges: List of tuples (source, target, attributes_dict) or (source, target)
            remove_edges: List of tuples (source, target) to remove
            
        Returns:
            New modified NetworkX graph
            
        Raises:
            IndexError: If graph_index is out of range
            ValueError: If invalid operations specified
        """
        if graph_index >= len(self.graphs) or graph_index < 0:
            raise IndexError(f"Graph index {graph_index} out of range [0, {len(self.graphs)-1}]")
        
        # Create a deep copy of the graph
        new_graph = self.graphs[graph_index].copy()
        
        # Add nodes
        if add_nodes:
            for item in add_nodes:
                if isinstance(item, tuple) and len(item) == 2:
                    node_id, attrs = item
                    if isinstance(attrs, dict):
                        new_graph.add_node(node_id, **attrs)
                    else:
                        new_graph.add_node(item)
                else:
                    new_graph.add_node(item)
        
        # Remove nodes
        if remove_nodes:
            for node in remove_nodes:
                if node in new_graph.nodes():
                    new_graph.remove_node(node)
        
        # Add edges
        if add_edges:
            for item in add_edges:
                if isinstance(item, tuple) and len(item) >= 2:
                    if len(item) == 3 and isinstance(item[2], dict):
                        source, target, attrs = item
                        new_graph.add_edge(source, target, **attrs)
                    else:
                        source, target = item[0], item[1]
                        new_graph.add_edge(source, target)
        
        # Remove edges
        if remove_edges:
            for source, target in remove_edges:
                if new_graph.has_edge(source, target):
                    new_graph.remove_edge(source, target)
        
        # Invalidate cache since we modified a graph
        self._aggregate_cache = None
        
        return new_graph
    
    def summary(self) -> Dict:
        """Get summary statistics of all graphs."""
        return {
            'num_graphs': len(self.graphs),
            'total_nodes': len(self.get_all_nodes()),
            'total_edges': len(self.get_all_edges()),
            'graphs_detail': [
                {
                    'nodes': len(g.nodes()),
                    'edges': len(g.edges()),
                    'density': nx.density(g)
                }
                for g in self.graphs
            ]
        }

if __name__ == "__main__":
    graphs = pickle.load(open('reports_processed_graphs.pkl', 'rb'))

    # Initialize manager
    manager = MedicalGraphManager(graphs)

    # Display summary
    print("\n1. AGGREGATE SUMMARY")
    print("-" * 70)
    summary = manager.summary()
    print(f"Number of graphs: {summary['num_graphs']}")
    print(f"Total unique nodes: {summary['total_nodes']}")
    print(f"Total unique edges: {summary['total_edges']}")
    print("\nPer-graph details:")

    #for i, detail in enumerate(summary['graphs_detail']):
    #    print(f"  Graph {i}: {detail['nodes']} nodes, {detail['edges']} edges, density: {detail['density']:.3f}")

    # Display all nodes and edges
    print("\n2. ALL AGGREGATED NODES")
    print("-" * 70)
    all_nodes = manager.get_all_nodes()
    #for node in sorted(all_nodes):
    #    print(f"  • {node}")

    print("\n3. ALL AGGREGATED EDGES")
    print("-" * 70)
    all_edges = manager.get_all_edges()
    #for source, target in sorted(all_edges):
    #    print(f"  • {source} → {target}")

    # Get node attributes across graphs
    print("\n4. NODE ATTRIBUTES ACROSS GRAPHS")
    print("-" * 70)
    attrs = manager.get_node_attributes('zone')
    print(f"Attribute found in graphs: {list(attrs.keys())}")
    #for graph_key, attr in attrs.items():
    #    print(f"  {graph_key}: {attr}")

    # Get edge attributes
    print("\n5. EDGE ATTRIBUTES ACROSS GRAPHS")
    print("-" * 70)
    edge_attrs = manager.get_edge_attributes('zone', 'tube')
    #print(f"Edge found in graphs: {list(edge_attrs.keys())}")
    #for graph_key, attrs in edge_attrs.items():
    #    print(f"  {graph_key}: {attrs}")


    # EDIT GRAPH 0 (Diabetes graph)
    print("\n6. EDITING GRAPH 0 ")
    print("-" * 70)
    print("Original nodes:", list(graphs[222162].nodes()))
    print("Original edges:", list(graphs[222162].edges()))

    # Create modified graph
    edited_graph = manager.edit_instance(
        graph_index=222162,
        add_nodes=[
            ('Neuropathy', {'type': 'symptom', 'severity': 'medium'}),
            ('Metformin', {'type': 'treatment', 'category': 'drug'})
        ],
        add_edges=[
            ('Diabetes', 'Neuropathy', {'relation': 'causes'}),
            ('Metformin', 'Diabetes', {'relation': 'manages'})
        ],
        remove_nodes=['zone'],
        remove_edges=[('zone', 'lung')]
    )

    print("\nModified nodes:", list(edited_graph.nodes()))
    print("Modified edges:", list(edited_graph.edges()))

    # Display the modified graph structure
    print("\n7. MODIFIED GRAPH STRUCTURE")
    print("-" * 70)
    print(f"Nodes in edited graph: {len(edited_graph.nodes())}")
    print(f"Edges in edited graph: {len(edited_graph.edges())}")
    for node in edited_graph.nodes():
        attrs = dict(edited_graph.nodes[node])
        print(f"  • {node}: {attrs}")

    print("\nEdges in modified graph:")
    for source, target in edited_graph.edges():
        attrs = dict(edited_graph.edges[source, target])
        print(f"  • {source} → {target}: {attrs}")
