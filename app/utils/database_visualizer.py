import logging
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DatabaseVisualizer:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
            self.logger.setLevel(logging.INFO)

    def visualize_database(self, df: pd.DataFrame) -> None:
        """Visualize database."""
        vectors = list(df['text_embeddings'])
        reduced_embeddings = self.reduce_embeddings_to_2d(vectors)
        self.plot_embeddings(df, reduced_embeddings)

    def reduce_embeddings_to_2d(self, vectors: list) -> np.ndarray:
        """Reduce embeddings to 2D using PCA."""
        pca = PCA(n_components=2)
        return pca.fit_transform(vectors)

    def plot_embeddings(self, df: pd.DataFrame, reduced_embeddings: np.ndarray) -> None:
        """Plot the reduced embeddings."""
        unique_docs = df['document_name'].unique()
        color_map = {doc: plt.cm.tab10(i % 10) for i, doc in enumerate(unique_docs)}
        colors = [color_map[doc] for doc in df['document_name']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.4)
        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")
        ax1.set_title("2D PCA Projection of Chunk-Embeddings")

        for doc in unique_docs:
            doc_indices = df['document_name'] == doc
            sns.kdeplot(
                x=reduced_embeddings[doc_indices, 0], 
                y=reduced_embeddings[doc_indices, 1], 
                ax=ax2, 
                cmap=sns.light_palette(color_map[doc], as_cmap=True), 
                fill=True, 
                alpha=0.4,
                warn_singular=False
            )

        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        ax2.set_title("KDE Density Estimation of Document Chunk-Embeddings")
        plt.show()
