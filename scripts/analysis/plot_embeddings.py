import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence


def plot_embeddings(
    embedding_2d: np.ndarray, 
    labels: np.ndarray, 
    class_map: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    point_size: int = 5,
    alpha: float = 0.8,
    title: str = 'UMAP Visualization with Class Distributions',
    show_ellipses: bool = True,
    show_centroids: bool = True,
    custom_colors: Optional[Sequence[Any]] = None,
    highlight_indices: Optional[np.ndarray] = None,
    file_path: Optional[str] = None,
    verbose: bool = True,
    return_object: bool = False,
     ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot 2D embeddings with class distributions, centroids, and statistics.
    
    Args:
        embedding_2d: 2D embeddings with shape (n_samples, 2)
        labels: Class labels starting from 1 (will be converted to 0-indexed)
        class_map: Dictionary mapping class IDs to class names
        figsize: Figure size as (width, height)
        point_size: Size of scatter points
        alpha: Transparency of scatter points
        title: Plot title
        show_ellipses: Whether to show 95% confidence ellipses
        show_centroids: Whether to show class centroids
        custom_colors: Optional custom colormap
        highlight_indices: Optional indices to highlight with larger markers
        file_path: Optional path to save figure
        verbose: If True, print class distribution statistics
        return_object: If True, return the matplotlib figure object
        ax: Optional matplotlib axes object to plot on
    
    Returns:
        The matplotlib figure
    """
    # Validations
    if embedding_2d.shape[0] != labels.shape[0]:
        raise ValueError(f"Embeddings and labels must have same length. Got {embedding_2d.shape[0]} and {labels.shape[0]}")
    
    if embedding_2d.shape[1] != 2:
        raise ValueError(f"Embeddings must be 2D. Got shape {embedding_2d.shape}")
    if class_map is not None:
        unique_labels = np.unique(labels)
        # Adjust expected keys based on whether class_map keys start at 1 or 0
        if min(class_map.keys()) == 1:
            expected_keys = set(unique_labels)
        else:
            expected_keys = set(unique_labels - 1)
        missing_keys = expected_keys - set(class_map.keys())
        if missing_keys:
            raise ValueError(f"Missing mapping for label(s): {missing_keys} in class_map")
    
    # Convert labels to 0-indexed for matplotlib compatibility
    if labels.min() != 0:
        labels_np = labels.copy() - 1
    else:
        labels_np = labels.copy()
    
    # Create or normalize class map
    if class_map is None:
        unique_classes = np.unique(labels)
        class_map = {c-1: f"Class {c}" for c in unique_classes}
    elif min(class_map.keys()) == 1:
        class_map = {k-1: v for k, v in class_map.items()}
    
    # Initialize plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set up colormap
    n_classes = len(np.unique(labels_np))
    cmap = mpl.colors.ListedColormap(custom_colors) if custom_colors else plt.get_cmap('tab10', n_classes)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_classes+0.5, 1), cmap.N)
    
    # Plot data points
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1],
        c=labels_np, 
        cmap=cmap, 
        norm=norm, 
        s=point_size, 
        alpha=alpha
    )
    
    # Highlight specific points if requested
    if highlight_indices is not None:
        plt.scatter(
            embedding_2d[highlight_indices, 0],
            embedding_2d[highlight_indices, 1],
            facecolors='none',
            edgecolors='black',
            s=point_size*4,
            linewidth=1.5,
            zorder=5,
            alpha=alpha
        )
    
    # Add class-specific visualizations (ellipses and centroids)
    unique_classes = np.unique(labels_np)
    for class_id in unique_classes:
        class_mask = (labels_np == class_id)
        class_points = embedding_2d[class_mask]
        
        if len(class_points) <= 1:
            continue  # Skip classes with too few points
            
        centroid = np.mean(class_points, axis=0)
        
        # Add distribution ellipse
        if show_ellipses:
            try:
                # Calculate 95% confidence ellipse based on covariance
                cov = np.cov(class_points, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * np.sqrt(eigenvalues) * 2
                
                ellipse = Ellipse(
                    xy=centroid,
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor=cmap(class_id),
                    facecolor='none',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(ellipse)
            except np.linalg.LinAlgError:
                print(f"Warning: Could not calculate ellipse for class {class_map[class_id]}")
        
        # Add centroid marker
        if show_centroids:
            plt.scatter(
                centroid[0],
                centroid[1],
                marker='*',
                s=300,
                color=cmap(class_id),
                edgecolor='black',
                linewidth=1.5,
                zorder=10
            )
            
            plt.annotate(
                f'{class_map[class_id]}',
                xy=centroid,
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
    
    # Add colorbar for class reference
    colorbar = plt.colorbar(scatter, ticks=np.arange(n_classes), label='Class')
    colorbar.ax.set_yticklabels([class_map[i] for i in range(n_classes)])
    
    # Add legend for visual elements
    legend_elements = []
    if show_ellipses:
        legend_elements.append(mpatches.Patch(
            facecolor='none', edgecolor='gray', linestyle='--', 
            label='Class Distribution (95%)'
        ))
    if show_centroids:
        legend_elements.append(plt.Line2D(
            [0], [0], marker='*', color='w', markerfacecolor='gray', 
            markersize=15, label='Class Centroid'
        ))
    if highlight_indices is not None:
        legend_elements.append(plt.Line2D(
            [0], [0], marker='o', color='w', markeredgecolor='black',
            markerfacecolor='none', markersize=10, label='Highlighted Points'
        ))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')
    
    # Finalize plot
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    # Save if requested
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Print distribution statistics
    if verbose:
        print("\nClass distribution statistics:")
        for class_id in unique_classes:
            count = np.sum(labels_np == class_id)
            percentage = (count / len(labels_np)) * 100
            print(f"{class_map[class_id]}: {count} samples ({percentage:.1f}%)")

    if return_object:
        return fig
