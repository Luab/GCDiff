#!/usr/bin/env env python3
"""
RadGraph Entity and Relation Extraction to NetworkX
Extracts entities and relations from radiology reports using RadGraph
and creates NetworkX graphs for each report.
"""

import pandas as pd
import networkx as nx
from radgraph import RadGraph
import json
from typing import Dict, List, Tuple
import pickle
from tqdm import tqdm

def extract_radgraph_entities_relations(report_text: str, radgraph_model) -> Tuple[List[Dict], List[Tuple]]:
    """
    Extract entities and relations from a single report using RadGraph.

    Args:
        report_text: The radiology report text
        radgraph_model: Initialized RadGraph model instance

    Returns:
        nodes: List of node dictionaries with entity information
        edges: List of tuples (source_id, target_id, relation_type)
    """
    # Get annotations from RadGraph
    annotations = radgraph_model([report_text])

    nodes = []
    edges = []

    # Process the first (and only) report in the batch
    report_key = list(annotations.keys())[0]
    report_data = annotations[report_key]

    # Extract entities as nodes
    entities = report_data.get('entities', {})

    for entity_id, entity_data in entities.items():
        node = {
            'id': entity_id,
            'tokens': entity_data.get('tokens', ''),
            'label': entity_data.get('label', ''),
            'start_ix': entity_data.get('start_ix', -1),
            'end_ix': entity_data.get('end_ix', -1)
        }
        nodes.append(node)

        # Extract relations as edges
        relations = entity_data.get('relations', [])
        for relation in relations:
            if len(relation) == 2:
                relation_type, target_id = relation
                edges.append((entity_id, target_id, relation_type))

    return nodes, edges

def create_networkx_graph(nodes: List[Dict], edges: List[Tuple]) -> nx.Graph:
    """
    Create a NetworkX directed graph from nodes and edges.

    Args:
        nodes: List of node dictionaries
        edges: List of edge tuples (source, target, relation_type)

    Returns:
        NetworkX DiGraph object
    """
    G = nx.Graph()

    # Add nodes with attributes
    for node in nodes:
        node_id = node['id']
        G.add_node(node_id, 
                   tokens=node['tokens'].lower(),
                   label=node['label'],
                   start_ix=node['start_ix'],
                   end_ix=node['end_ix'])

    # Add edges with relation type as attribute
    for source, target, relation_type in edges:
        G.add_edge(source, target, relation=relation_type)

    mapping = nx.get_node_attributes(G, "tokens")
    G_ = nx.relabel_nodes(G, mapping)
 
    return G_

def process_csv_reports(input_csv: str, output_csv: str = None, 
                        model_type: str = "modern-radgraph-xl"):
    """
    Process all reports in a CSV file and extract RadGraph entities/relations.

    Args:
        input_csv: Path to input CSV file with 'report' column
        output_csv: Optional path to save results (default: adds '_radgraph' suffix)
        model_type: RadGraph model type to use
    """
    # Load CSV
    print(f"Loading CSV from {input_csv}...")
    df = pd.read_csv(input_csv)

    if 'report' not in df.columns:
        raise ValueError("CSV must contain a 'report' column")

    # Initialize RadGraph model
    print(f"Initializing RadGraph model: {model_type}...")
    radgraph = RadGraph(model_type=model_type)

    # Process each report
    graphs = []
    nodes_list = []
    edges_list = []

    print(f"Processing {len(df)} reports...")
    for idx, row in tqdm(df.iterrows()):
        report_text = row['report']

        if pd.isna(report_text) or str(report_text).strip() == '':
            print(f"  Row {idx}: Empty report, skipping")
            graphs.append(None)
            nodes_list.append([])
            edges_list.append([])
            continue

        try:
            # Extract entities and relations
            nodes, edges = extract_radgraph_entities_relations(report_text, radgraph)

            # Create NetworkX graph
            graph = create_networkx_graph(nodes, edges)

            graphs.append(graph)
            nodes_list.append(nodes)
            edges_list.append(edges)

            print(f"  Row {idx}: Extracted {len(nodes)} entities and {len(edges)} relations")

        except Exception as e:
            print(f"  Row {idx}: Error processing report - {str(e)}")
            graphs.append(None)
            nodes_list.append([])
            edges_list.append([])

    # Add results to dataframe
    df['radgraph_nodes'] = nodes_list
    df['radgraph_edges'] = edges_list
    df['radgraph_graph'] = graphs
    df['num_entities'] = [len(nodes) for nodes in nodes_list]
    df['num_relations'] = [len(edges) for edges in edges_list]

    # Save results
    if output_csv is None:
        output_csv = input_csv.rsplit('.', 1)[0] + '_radgraph.csv'

    # Save CSV (note: NetworkX graphs will be converted to string representation)
    print(f"Saving results to {output_csv}...")
    df_to_save = df.copy()
    df_to_save['radgraph_graph'] = df_to_save['radgraph_graph'].apply(
        lambda g: f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}" if g else "None"
    )
    df_to_save.to_csv(output_csv, index=False)

    # Save graphs as pickle for later use
    graphs_pickle = output_csv.rsplit('.', 1)[0] + '_graphs.pkl'
    print(f"Saving NetworkX graphs to {graphs_pickle}...")
    with open(graphs_pickle, 'wb') as f:
        pickle.dump(graphs, f)

    print("\nProcessing complete!")
    print(f"  Total reports: {len(df)}")
    print(f"  Total entities extracted: {df['num_entities'].sum()}")
    print(f"  Total relations extracted: {df['num_relations'].sum()}")

    return df, graphs

# Example usage functions
def example_visualize_graph(graph: nx.Graph):
    """
    Print a text representation of the graph.
    """
    if graph is None or graph.number_of_nodes() == 0:
        print("Empty graph")
        return

    print(f"\nGraph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print("\nNodes:")
    for node_id, attrs in graph.nodes(data=True):
        print(f"  {node_id}: {attrs['tokens']} ({attrs['label']})")

    print("\nEdges:")
    for source, target, attrs in graph.edges(data=True):
        source_token = graph.nodes[source]['tokens']
        target_token = graph.nodes[target]['tokens']
        relation = attrs['relation']
        print(f"  {source_token} --[{relation}]--> {target_token}")

def example_graph_statistics(graphs: List[nx.Graph]):
    """
    Print statistics about the extracted graphs.
    """
    valid_graphs = [g for g in graphs if g is not None and g.number_of_nodes() > 0]

    print("\n=== Graph Statistics ===")
    print(f"Total graphs: {len(graphs)}")
    print(f"Non-empty graphs: {len(valid_graphs)}")

    if valid_graphs:
        avg_nodes = sum(g.number_of_nodes() for g in valid_graphs) / len(valid_graphs)
        avg_edges = sum(g.number_of_edges() for g in valid_graphs) / len(valid_graphs)
        print(f"Average nodes per graph: {avg_nodes:.2f}")
        print(f"Average edges per graph: {avg_edges:.2f}")

# Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv> [output_csv] [model_type]")
        print("Example: python script.py reports.csv reports_processed.csv modern-radgraph-xl")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    model_type = sys.argv[3] if len(sys.argv) > 3 else "modern-radgraph-xl"

    # Process the CSV
    df, graphs = process_csv_reports(input_csv, output_csv, model_type)

    # Show statistics
    example_graph_statistics(graphs)

    # Show example of first non-empty graph
    for idx, graph in enumerate(graphs):
        if graph is not None and graph.number_of_nodes() > 0:
            print(f"\n=== Example Graph (Row {idx}) ===")
            example_visualize_graph(graph)
            break