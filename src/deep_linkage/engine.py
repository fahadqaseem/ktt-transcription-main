# engine.py
import numpy as np
import pandas as pd
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from .config import DeepERConfig
from .schema import flatten_record


class DeepLinkageEngine:
    def __init__(self, config: DeepERConfig):
        self.config = config
        self.graph = nx.Graph()
        self.model = None

    def load_model(self, model_path: str = None):
        """Loads fine-tuned historical embedding model."""
        path = model_path or self.config.custom_model_out_path
        print(f"Loading inference embedding model from: {path}")
        self.model = SentenceTransformer(path)

    def encode_records(self, df: pd.DataFrame) -> np.ndarray:
        """Serializes rows via shared schema.py and embeds them into 384d vectors."""
        if self.model is None:
            self.load_model()
        print("Flattening master records via standardized schema...")
        sentences = [flatten_record(row) for _, row in df.fillna("").iterrows()]

        print(f"Encoding {len(sentences)} records into dense vectors...")
        embeddings = self.model.encode(
            sentences, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings

    def build_candidate_graph(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        Builds weighted candidate graph using FAISS IndexFlatIP (Cosine Similarity).
        """
        print("Building FAISS candidate index...")
        num_records, dim = embeddings.shape

        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        top_k = self.config.top_k_neighbors
        print(
            f"Searching top-{top_k} nearest neighbors for {num_records} records..."
        )
        similarities, indices = index.search(embeddings, k=top_k)

        self.graph.clear()
        for idx in range(num_records):
            self.graph.add_node(idx, record_id=df.index[idx])

        edge_count = 0
        for i in range(num_records):
            for j_rank in range(1, top_k):
                j = indices[i, j_rank]
                sim = float(similarities[i, j_rank])

                if sim >= self.config.similarity_threshold and i < j:
                    self.graph.add_edge(i, j, weight=sim)
                    edge_count += 1

        print(
            f"Candidate graph constructed: {num_records} nodes, {edge_count} weighted edges."
        )

    def _prune_open_triangles(self):
        """
        Bridge-Pruning Guardrail:
        In any path A - B - C where edge (A, C) does not exist,
        drop the weaker of (A, B) or (B, C) if its weight is near threshold boundary.
        """
        edges_to_remove = set()
        for node_b in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node_b))
            if len(neighbors) < 2:
                continue

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    node_a, node_c = neighbors[i], neighbors[j]

                    if not self.graph.has_edge(node_a, node_c):
                        w_ab = self.graph[node_a][node_b]["weight"]
                        w_bc = self.graph[node_b][node_c]["weight"]

                        weaker_edge = (
                            (node_a, node_b) if w_ab < w_bc else (node_b, node_c)
                        )
                        min_weight = min(w_ab, w_bc)

                        if min_weight < (
                            self.config.similarity_threshold + 0.04
                        ):
                            edges_to_remove.add(weaker_edge)

        if edges_to_remove:
            print(
                f"✂️ Severing {len(edges_to_remove)} unclosed bridge edges to prevent chaining..."
            )
            self.graph.remove_edges_from(edges_to_remove)

    def resolve_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns person_id clusters using Weighted Louvain Modularity Optimization.
        """
        if self.config.prune_weak_bridges:
            self._prune_open_triangles()

        print(
            f"Running Louvain community detection (resolution={self.config.louvain_resolution})..."
        )
        communities = nx.community.louvain_communities(
            self.graph,
            weight="weight",
            resolution=self.config.louvain_resolution,
            seed=42,
        )

        person_id_map = {}
        for cluster_id, node_set in enumerate(communities, start=1):
            for node_idx in node_set:
                person_id_map[node_idx] = f"PID_{cluster_id:06d}"

        result_df = df.copy()
        result_df["person_id"] = [
            person_id_map.get(idx, f"PID_UNLINKED_{idx}")
            for idx in range(len(result_df))
        ]

        num_clusters = len(communities)
        multi_record_clusters = sum(1 for c in communities if len(c) > 1)
        print(
            f"✅ Resolved {len(df)} records into {num_clusters} distinct entities."
        )
        print(
            f"   ({multi_record_clusters} entities span multiple historical years)"
        )
        return result_df