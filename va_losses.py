"""
[TESIS] Archivo nuevo (no existe en timnet original).
CCC-based loss functions for Valence-Arousal regression.
Based on Concordance Correlation Coefficient (Lin, 1989).
"""
import tensorflow as tf


def ccc_loss(y_true, y_pred):
    """
    Concordance Correlation Coefficient loss for a single dimension.
    CCC = 2 * cov(y, y_hat) / (var(y) + var(y_hat) + (mean(y) - mean(y_hat))^2)
    Loss = 1 - CCC

    Args:
        y_true: (B,) ground truth values
        y_pred: (B,) predicted values
    Returns:
        Scalar loss in [0, 2]
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mu_true = tf.reduce_mean(y_true)
    mu_pred = tf.reduce_mean(y_pred)
    var_true = tf.math.reduce_variance(y_true)
    var_pred = tf.math.reduce_variance(y_pred)
    covar = tf.reduce_mean((y_true - mu_true) * (y_pred - mu_pred))
    ccc = (2.0 * covar) / (var_true + var_pred + tf.square(mu_true - mu_pred) + 1e-8)
    return 1.0 - ccc

def combined_ccc_loss(y_true, y_pred):
    """
    Combined CCC loss for valence and arousal.
    y_true and y_pred have shape (B, 2) where column 0 = valence, column 1 = arousal.

    Returns:
        Scalar loss = 0.5 * CCC_loss_valence + 0.5 * CCC_loss_arousal
    """
    loss_v = ccc_loss(y_true[:, 0], y_pred[:, 0])
    loss_a = ccc_loss(y_true[:, 1], y_pred[:, 1])
    return 0.5 * loss_v + 0.5 * loss_a


def ccc_metric_valence(y_true, y_pred):
    """CCC for valence (column 0). Returns CCC value (higher is better)."""
    return 1.0 - ccc_loss(y_true[:, 0], y_pred[:, 0])


def ccc_metric_arousal(y_true, y_pred):
    """CCC for arousal (column 1). Returns CCC value (higher is better)."""
    return 1.0 - ccc_loss(y_true[:, 1], y_pred[:, 1])
