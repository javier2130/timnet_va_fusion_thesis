"""
[TESIS] Archivo nuevo (no existe en timnet).
Valence-Arousal anchor coordinates for emotion mapping.
Based on psychological findings from Russell (1980) and Zhou et al. (2024).
"""
import numpy as np

# V-A anchors: emotion_name -> (valence, arousal)
# Values from Zhou et al., 2024 "Learning Arousal-Valence Representation
# from Categorical Emotion Labels of Speech"
VA_ANCHORS = {
    "angry":      (-0.51,  0.59),
    "happy":      ( 0.81,  0.51),
    "sad":        (-0.63, -0.27),
    "fear":       (-0.64,  0.60),
    "fearful":    (-0.64,  0.60),
    "neutral":    ( 0.00,  0.00),
    "boredom":    (-0.65, -0.62),
    "disgust":    (-0.60,  0.35),
    "disgusted":  (-0.60,  0.35),
    "excited":    ( 0.62,  0.75),
    "frustrated": (-0.64,  0.52),
    "surprise":   ( 0.40,  0.67),
    "surprised":  ( 0.40,  0.67),
    "calm":       ( 0.20, -0.80),
    "contempt":   (-0.80,  0.20),
}


def get_anchor_matrix(class_labels):
    """
    Build a (C, 2) anchor matrix from class label names.

    Args:
        class_labels: tuple of emotion names, e.g. ("angry", "happy", "sad")
    Returns:
        np.ndarray of shape (C, 2) with [valence, arousal] per class
    """
    anchors = []
    for label in class_labels:
        if label not in VA_ANCHORS:
            raise ValueError(f"No V-A anchor defined for emotion '{label}'")
        anchors.append(VA_ANCHORS[label])
    return np.array(anchors, dtype=np.float32)


def labels_to_va_targets(y_onehot, class_labels):
    """
    Convert one-hot categorical labels to V-A target coordinates.

    Args:
        y_onehot: np.ndarray of shape (N, C), one-hot encoded labels
        class_labels: tuple of emotion names, length C
    Returns:
        np.ndarray of shape (N, 2) with [valence, arousal] per sample
    """
    anchor_matrix = get_anchor_matrix(class_labels)  # (C, 2)
    label_indices = np.argmax(y_onehot, axis=1)      # (N,)
    return anchor_matrix[label_indices]                # (N, 2)
