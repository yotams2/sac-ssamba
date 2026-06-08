import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings

class SACDebugger:
    """
    A comprehensive diagnostics and visualization suite for debugging continuous 
    contrastive learning dynamics, the feature space, and the latent manifold.
    """
    def __init__(self, feature_names=None):
        if feature_names is None:
            self.feature_names = ['μ_F0', 'HNR', 'μ_Centroid', 'σ_Flux', 'ZCR']
        else:
            self.feature_names = feature_names

    def _to_numpy(self, tensor):
        """Safely detaches and moves a tensor to CPU NumPy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def plot_feature_distributions(self, c):
        """
        Plots feature distributions and their correlation matrix.
        
        Args:
            c: [B, 5] normalized feature tensor.
            
        Returns:
            fig_hist: A 1x5 grid of histograms.
            fig_corr: A 5x5 heatmap of the Pearson correlation matrix.
        """
        c_np = self._to_numpy(c)
        B, num_features = c_np.shape

        # Guard against mismatch between configured names and actual tensor width
        if len(self.feature_names) != num_features:
            names = (self.feature_names + [f'Feat {i}' for i in range(num_features)])[:num_features]
        else:
            names = self.feature_names
        
        # 1. Feature Histograms
        fig_hist, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
        if num_features == 1:
            axes = [axes]
            
        for i in range(num_features):
            sns.histplot(c_np[:, i], kde=True, ax=axes[i], color='skyblue', stat="density")
            axes[i].set_title(names[i])
            axes[i].set_xlim(-1.1, 1.1)
        fig_hist.tight_layout()
        
        # 2. Correlation Matrix Heatmap
        # Handle zero variance warning when calculating correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(c_np, rowvar=False)
            
        # Replace NaNs with 0s (happens if a feature has zero variance)
        corr = np.nan_to_num(corr)
            
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f",
                    xticklabels=names, yticklabels=names, ax=ax_corr)
        ax_corr.set_title("Pearson Correlation Matrix")
        fig_corr.tight_layout()
        
        return fig_hist, fig_corr

    def plot_kernel_and_similarity(self, s, w):
        """
        Plots the distribution of raw similarity, weights, and their relationship.
        
        Args:
            s: [B, B] raw cosine similarity matrix (temperature-scaled).
            w: [B, B] raw Gaussian kernel weights (before row-normalization).
            
        Returns:
            fig: A 1x3 grid containing the weight histogram, similarity histogram, 
                 and a scatter plot of weight vs similarity.
        """
        s_np = self._to_numpy(s)
        w_np = self._to_numpy(w)
        B = s_np.shape[0]
        
        if B < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Batch size too small for off-diagonal stats", ha="center")
            return fig
            
        # Extract off-diagonal elements
        mask = ~np.eye(B, dtype=bool)
        s_off = s_np[mask]
        w_off = w_np[mask]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.histplot(w_off, bins=50, ax=axes[0], color='green', kde=True, stat="density")
        axes[0].set_title("Histogram of w_ij (Off-diagonal)")
        axes[0].set_xlabel("w_ij (Kernel Weight)")
        
        sns.histplot(s_off, bins=50, ax=axes[1], color='purple', kde=True, stat="density")
        axes[1].set_title("Histogram of s_ij (Off-diagonal)")
        axes[1].set_xlabel("s_ij (Latent Similarity)")
        
        axes[2].scatter(w_off, s_off, alpha=0.3, s=5, c='orange')
        axes[2].set_title("Latent Similarity vs Kernel Weight")
        axes[2].set_xlabel("w_ij (Kernel Weight)")
        axes[2].set_ylabel("s_ij (Latent Similarity)")
        
        fig.tight_layout()
        return fig

    def compute_alignment_uniformity(self, z_norm, w):
        """
        Computes Manifold Health Metrics.
        
        Args:
            z_norm: [B, proj_dim] L2-normalized latent embeddings.
            w: [B, B] raw Gaussian kernel weights.
            
        Returns:
            uniformity: Scalar log E[exp(-2 * ||z_i - z_j||^2)].
            alignment_top10: Average cosine similarity of pairs in top 10% of w_ij.
        """
        z_norm_np = self._to_numpy(z_norm)
        w_np = self._to_numpy(w)
        B = z_norm_np.shape[0]
        
        if B < 2:
            return 0.0, 0.0
            
        # Mask off-diagonal
        mask = ~np.eye(B, dtype=bool)
        w_off = w_np[mask]
        
        # 1. Uniformity
        # Calculate pairwise squared Euclidean distance: ||z_i - z_j||^2
        # Since z is L2 normalized: ||z_i - z_j||^2 = 2 - 2*(z_i dot z_j)
        sim = z_norm_np @ z_norm_np.T
        z_dist_sq = 2.0 - 2.0 * sim
        z_dist_sq_off = z_dist_sq[mask]
        
        # Uniformity: log E[exp(-2 * ||z_i - z_j||^2)]
        uniformity = np.log(np.mean(np.exp(-2.0 * z_dist_sq_off)))
        
        # 2. Alignment Top 10%
        # Find threshold for top 10% weights
        top10_threshold = np.percentile(w_off, 90)
        top10_mask = w_off >= top10_threshold
        
        sim_off = sim[mask]
        
        if np.any(top10_mask):
            alignment_top10 = np.mean(sim_off[top10_mask])
        else:
            alignment_top10 = 0.0
            
        return float(uniformity), float(alignment_top10)

    def plot_latent_manifold(self, z, c):
        """
        Projects latent embeddings to 2D and color-maps by acoustic features.
        
        Args:
            z: [B, proj_dim] latent embeddings.
            c: [B, 5] normalized feature tensor.
            
        Returns:
            fig: A 1x5 grid of scatter plots (2D projection).
        """
        z_np = self._to_numpy(z)
        c_np = self._to_numpy(c)
        B, num_features = c_np.shape

        # Guard against mismatch between configured names and actual tensor width
        if len(self.feature_names) != num_features:
            names = (self.feature_names + [f'Feat {i}' for i in range(num_features)])[:num_features]
        else:
            names = self.feature_names
        
        if B < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Batch size too small for manifold projection", ha="center")
            return fig

        # Attempt to use UMAP, fallback to t-SNE
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            method_name = "UMAP"
        except ImportError:
            # Perplexity must be less than number of samples
            perplexity = min(30, max(1, B - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            method_name = "t-SNE"
            
        # Add slight noise to avoid TSNE error if all points are identical
        if np.all(z_np == z_np[0]):
            z_np += np.random.normal(0, 1e-4, z_np.shape)
            
        z_2d = reducer.fit_transform(z_np)
        
        fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
        if num_features == 1:
            axes = [axes]
            
        for i in range(num_features):
            sc = axes[i].scatter(z_2d[:, 0], z_2d[:, 1], c=c_np[:, i], cmap='viridis', s=20, alpha=0.8)
            axes[i].set_title(f"{method_name} colored by\n{names[i]}")
            axes[i].axis('off')
            fig.colorbar(sc, ax=axes[i])
            
        fig.tight_layout()
        return fig
