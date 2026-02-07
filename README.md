# GCDiff-X: Graph-Conditioned Diffusion for Explanations

This repository implements graph-conditioned diffusion for chest X-ray counterfactual generation and evaluation.

---

## 0. Graph processing pipeline

Before training, reports are turned into graphs and graph embeddings. All steps live under `graph_processing/`.

**Step 0a – NER and graph extraction** (RadGraph → NetworkX):

From a CSV that has a `report` column (e.g. raw CheXpert reports), extract entities and relations with RadGraph and build one NetworkX graph per report.

```bash
python graph_processing/NER_extraction.py <input_csv> [output_csv] [model_type]
# Example:
python graph_processing/NER_extraction.py reports.csv reports_processed.csv modern-radgraph-xl
```

- **Input:** CSV with `report` column.
- **Output:**  
  - CSV with added columns: `radgraph_nodes`, `radgraph_edges`, `radgraph_graph` (summary), `num_entities`, `num_relations`.  
  - Pickle of NetworkX graphs: `<output_csv_basename>_graphs.pkl` (e.g. `reports_processed_graphs.pkl`).

**Step 0b – Graph embeddings** (Graph2Vec, 768-d):

Train Graph2Vec on the extracted graphs and encode them to fixed-size vectors. Uses `graph_processing/graph_embedding.py` (SimpleGraph2VecEmbedder from karateclub).

1. Load graphs from `reports_processed_graphs.pkl`, fit the embedder, save it (e.g. `graph2vec_embeddings.pkl`).
2. Encode all graphs and save 768-d vectors (e.g. `new_embeddings_768.pkl`).

These pickles are what training and evaluation expect (see `graph_embedding.py` `__main__` for exact paths and optional HDF5 expansion).

**Step 0c – Graph editing** (counterfactual edits at evaluation time):

`graph_processing/graph_editing.py` defines `MedicalGraphManager`: it takes the list of NetworkX graphs and supports instance-level edits (e.g. add/remove/relabel nodes or relations) and generation of modified graphs. Used during evaluation to produce “edited” graphs for counterfactual generation (graph-based and hybrid methods).

**Summary:**  
`reports CSV` → **NER_extraction.py** → `reports_processed.csv` + `reports_processed_graphs.pkl` → **graph_embedding.py** → `graph2vec_embeddings.pkl` + `new_embeddings_768.pkl`. Training uses the CSV, graphs pickle, and 768-d embeddings; evaluation also uses the embedder and graph manager for edits.

---

## 1. Training the model

**Prerequisites**

- `reports_processed.csv` (report–image metadata)
- `new_embeddings_768.pkl` (graph embeddings, 768-d)
- CheXpert images at `data/PNG/PNG` (or set paths accordingly)

**Run training (ControlNet):**

```bash
source venv/bin/activate
./train_controlnet.sh
```

Checkpoints are saved under `checkpoints/controlnet-linear/` (per-epoch and `final/`). Default: 10 epochs, batch size 16, gradient accumulation 4, LR 1e-4.

---

## 2. Evaluating

**Pathology sweep** (graph, text, and hybrid methods over 15 CheXpert pathologies):

```bash
./run_pathology_sweep.sh
```

Output goes to `outputs/sweep_<timestamp>/`. Each method (e.g. `graph_fixed`, `text_fixed`, `hybrid_fixed`) has a subfolder per pathology with `results.json`.

**Optional environment variables:** `METHODS`, `PATHOLOGIES`, `OUTPUT_BASE`, `CONTROLNET_PATH`, `EMBEDDER_PATH`, `GRAPHS_PATH`, `IMAGE_ROOT`, `BATCH_SIZE`, `GUIDANCE_SCALE`, `NUM_INFERENCE_STEPS`, `SKIP`, `CLASSIFIER_MODELS`, `TEMPLATE_SET`.

Example (subset of methods):

```bash
METHODS="graph_fixed text_fixed" ./run_pathology_sweep.sh
```

**Optional: FRD (Frechet Radiomics Distance)** on a sweep folder:

```bash
./evaluate_frd.sh /path/to/outputs/sweep_<timestamp>
```

Writes `frd_results.csv` inside that sweep folder.

---

## 3. Generating tables and CSV

**From a sweep directory** (e.g. `outputs/sweep_20260126_123642/`):

**Step 3a – Sweep analysis (classifier metrics, comparison tables):**

```bash
python analyze_sweep.py outputs/sweep_20260126_123642/ [--output <dir>] [--no-plots]
```

Produces in the sweep directory (or `--output`):

- `sweep_comparison.csv` – per-method, per-pathology metrics
- `sweep_summary.csv` – per-method aggregates (mean ± std)
- `plots/` – comparison figures (unless `--no-plots`)

**Step 3b – LaTeX-ready tables** (uses sweep CSVs and optionally FRD):

```bash
python scripts/generate_latex_tables.py \
  --sweep outputs/sweep_20260126_123642/sweep_comparison.csv \
  [--frd outputs/sweep_20260126_123642/frd_results.csv] \
  [--output-dir <dir>]
```

If you pass a sweep directory instead of a file, the script looks for `sweep_comparison.csv` and `frd_results.csv` inside it:

```bash
python scripts/generate_latex_tables.py --sweep outputs/sweep_20260126_123642/ --output-dir tables/
```

Outputs CSV files suitable for inclusion in papers (e.g. `table_frd.csv`, classifier metric tables in the chosen `--output-dir`).

**Template ablation** (if you ran sweeps with different `TEMPLATE_SET`):

```bash
python analyze_template_ablation.py /path/to/ablation_output_dir/ [--output <dir>]
```

---


Graphs and 768-d embeddings are aligned 1:1 with the CSV. Images are resized to 512×512 for training and evaluation.
