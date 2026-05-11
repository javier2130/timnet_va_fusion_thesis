"""
[TESIS] Archivo nuevo (no existe en timnet original).
Visualization of Valence-Arousal predictions on the circumplex model.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from va_config import get_anchor_matrix

def plot_va_circumplex(va_true, va_pred, labels, class_labels, dataset_name, save_path):
    """
    Plot V-A predictions on the circumplex model (before vs after).

    Args:
        va_true: (N, 2) anchor-based V-A coordinates
        va_pred: (N, 2) model-predicted V-A coordinates
        labels: (N,) integer class labels
        class_labels: tuple of emotion names
        dataset_name: string for title
        save_path: file path to save PNG
    """
    num_classes = len(class_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    anchor_matrix = get_anchor_matrix(class_labels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, title in [
        (axes[0], va_true, "V-A Anchors (Ground Truth)"),
        (axes[1], va_pred, "V-A Predictions (TIM-Net)")
    ]:
        # Draw quadrant lines
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

        # Draw unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=0.5)

        # Scatter by emotion
        for i, label in enumerate(class_labels):
            mask = labels == i
            ax.scatter(data[mask, 0], data[mask, 1],
                       c=[colors[i]], label=label, s=20, alpha=0.6, edgecolors='none')

        # Mark anchors
        for i, label in enumerate(class_labels):
            ax.scatter(anchor_matrix[i, 0], anchor_matrix[i, 1],
                       c=[colors[i]], marker='X', s=150, edgecolors='black', linewidths=1, zorder=5)
            ax.annotate(label, (anchor_matrix[i, 0], anchor_matrix[i, 1]),
                        textcoords="offset points", xytext=(8, 8), fontsize=8, fontweight='bold')

        # Quadrant labels
        ax.text(0.5, 0.5, 'Happy/Excited', transform=ax.transAxes, fontsize=8, alpha=0.3, ha='center')
        ax.text(0.15, 0.5, 'Angry/Fear', transform=ax.transAxes, fontsize=8, alpha=0.3, ha='center')
        ax.text(0.15, 0.15, 'Sad/Bored', transform=ax.transAxes, fontsize=8, alpha=0.3, ha='center')
        ax.text(0.5, 0.15, 'Calm/Relaxed', transform=ax.transAxes, fontsize=8, alpha=0.3, ha='center')

        ax.set_xlabel('Valence  (unpleasant ↔ pleasant)', fontsize=12)
        ax.set_ylabel('Arousal  (calm ↔ active)', fontsize=12)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(markerscale=1.5, fontsize=8, loc='upper left')

    plt.suptitle(f'Valence-Arousal Circumplex — {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"V-A circumplex saved: {save_path}")


def plot_va_circumplex_predictions(va_pred, labels, class_labels, dataset_name, save_path):
    """
    Plot V-A regression-head predictions on Russell's circumplex (single panel).

    Predictions come from Dense(2, activation='tanh') of VAFusionLayer, so each
    axis is bounded in [-1, 1] (square, not circle). Anchors (Russell emotions)
    overlaid as X markers.

    Args:
        va_pred: (N, 2) model-predicted V-A coordinates
        labels: (N,) integer class labels
        class_labels: tuple of emotion names
        dataset_name: string for title
        save_path: file path to save PNG
    """
    num_classes = len(class_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    anchor_matrix = get_anchor_matrix(class_labels)

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta),
            color='dimgray', linewidth=1.0, linestyle='-', alpha=0.7,
            label='Unit circle')

    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1],
            color='red', linewidth=0.8, linestyle=':', alpha=0.4,
            label='tanh bounds [-1, 1]²')

    for i, label in enumerate(class_labels):
        mask = labels == i
        ax.scatter(va_pred[mask, 0], va_pred[mask, 1],
                   c=[colors[i]], label=label, s=20, alpha=0.5, edgecolors='none')

    for i, label in enumerate(class_labels):
        ax.scatter(anchor_matrix[i, 0], anchor_matrix[i, 1],
                   c=[colors[i]], marker='X', s=220, edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(label, (anchor_matrix[i, 0], anchor_matrix[i, 1]),
                    textcoords="offset points", xytext=(10, 10), fontsize=10, fontweight='bold')

    ax.text(0.75, 0.92, 'Happy/Excited', transform=ax.transAxes, fontsize=9, alpha=0.45, ha='center')
    ax.text(0.25, 0.92, 'Angry/Fear', transform=ax.transAxes, fontsize=9, alpha=0.45, ha='center')
    ax.text(0.25, 0.08, 'Sad/Bored', transform=ax.transAxes, fontsize=9, alpha=0.45, ha='center')
    ax.text(0.75, 0.08, 'Calm/Relaxed', transform=ax.transAxes, fontsize=9, alpha=0.45, ha='center')

    ax.set_xlabel('Valence  (unpleasant ↔ pleasant)', fontsize=12)
    ax.set_ylabel('Arousal  (calm ↔ active)', fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(
        f'V-A Circumplex — {dataset_name}\nVAFusionLayer regression head (tanh-bounded), Russell anchors as X',
        fontsize=12, fontweight='bold'
    )
    ax.legend(markerscale=1.5, fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"V-A circumplex (predictions) saved: {save_path}")
