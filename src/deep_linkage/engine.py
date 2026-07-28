import pandas as pd
import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from .config import DeepERConfig


class DeepLinkageEngine:
    def __init__(self, config: DeepERConfig):
        self.config = config
        print(f"Loading Neural Network: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)

    def _flatten_schema(self, df: pd.DataFrame) -> list[str]:
        """Flatten record key-value pairs into structured text sentences for transformer input."""
        sentences = []
        # Columns to ignore during string serialization if they exist
        ignore_cols = {"id", "person_id", "ground_truth_id", "cluster_id"}

        for _, row in df.iterrows():
            record_parts = []
            for col, val in row.items():
                if col.lower() not in ignore_cols and pd.notna(val):
                    record_parts.append(f"{col}: {val}")
            sentence = " | ".join(record_parts)
            sentences.append(sentence)

        return sentences

    def _encode_records(self, sentences: list[str]) -> np.ndarray:
        """Encodes text representations into normalized vector embeddings."""
        print("Vectorizing records...")
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def _build_faiss_index(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Builds a FAISS index and performs K-Nearest Neighbor search."""
        print("Building FAISS index for high-speed blocking...")
        dimension = embeddings.shape[1]

        # Inner Product index (equivalent to Cosine Similarity when embeddings are L2 normalized)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Search for top-K neighbors
        # Convert similarity score (1.0 = identical) to Cosine Distance (0.0 = identical)
        similarities, indices = index.search(embeddings, self.config.top_k)
        distances = 1.0 - similarities

        return distances, indices

    def _cluster_entities(
        self, df: pd.DataFrame, distances: np.ndarray, indices: np.ndarray
    ) -> pd.DataFrame:
        """Constructs an adjacency graph from FAISS matches and applies connected component clustering."""
        print("Clustering graph components...")
        G = nx.Graph()

        # Add a node for every record row index
        for i in range(len(df)):
            G.add_node(i)

        # Connect nodes with edges if distance <= threshold
        for i in range(len(df)):
            for dist, neighbor_idx in zip(distances[i], indices[i]):
                if neighbor_idx != -1 and neighbor_idx != i:
                    if dist <= self.config.distance_threshold:
                        G.add_edge(i, neighbor_idx, weight=float(dist))

        # Output DataFrame setup with explicit object/string dtype initialization
        output_df = df.copy()
        output_df["person_id"] = "unclustered"

        # Assign cluster labels based on connected components
        for cluster_id, component in enumerate(nx.connected_components(G)):
            cluster_name = f"cluster_{cluster_id}"
            for node in component:
                output_df.at[node, "person_id"] = cluster_name

        return output_df

    def run_pipeline(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Executes full entity linkage end-to-end."""
        sentences = self._flatten_schema(input_df)
        embeddings = self._encode_records(sentences)
        distances, indices = self._build_faiss_index(embeddings)
        linked_df = self._cluster_entities(input_df, distances, indices)
        return linked_df