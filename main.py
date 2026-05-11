"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
Modernized for TensorFlow >= 2.16 / Keras 3
"""
import numpy as np
import os
import tensorflow as tf
from Model import TIMNET_Model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
parser.add_argument('--data', type=str, default='EMODB')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--save_best_only', action='store_true', default=False, help='Save only best weights per fold (monitor val_loss)')
# ===== [TESIS] hiperparámetros de pesos para loss multi-task V-A =====
parser.add_argument('--lambda_ce', type=float, default=0.33, help='Weight for classification loss')
parser.add_argument('--lambda_v', type=float, default=0.33, help='Weight for valence CCC loss')
parser.add_argument('--lambda_a', type=float, default=0.33, help='Weight for arousal CCC loss')
# ===== [/TESIS] =====
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

if args.data == "IEMOCAP" and args.dilation_size != 10:
    args.dilation_size = 10

# --- Modern GPU configuration (replaces tf.compat.v1.Session) ---
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print(f"###gpus:{gpus}")

CLASS_LABELS_finetune = ("angry", "fear", "happy", "neutral", "sad")
CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
SAVEE_CLASS_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise")#rav
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")#emovo
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS}

data = np.load("./MFCC/" + args.data + ".npy", allow_pickle=True).item()
x_source = data["x"]
y_source = data["y"]
CLASS_LABELS = CLASS_LABELS_dict[args.data]


model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
if args.mode == "train":
    model.train(x_source, y_source)
elif args.mode == "test":
    x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)# x_feats and y_labels are test datas for t-sne
elif args.mode == "tsne":
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import datetime

    raw_feats, learned_feats, labels = model.train_tsne(x_source, y_source)

    perplexity = 30
    if perplexity >= labels.shape[0]:
        perplexity = max(5, labels.shape[0] // 3)

    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne_raw = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embed_raw = tsne_raw.fit_transform(raw_feats)

    tsne_learned = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embed_learned = tsne_learned.fit_transform(learned_feats)

    num_classes = len(CLASS_LABELS)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, embed, title in [
        (axes[0], embed_raw, "Raw MFCC (T×39 dims, flattened)"),
        (axes[1], embed_learned, "TIM-Net Features — penultimate layer (39-dim)")
    ]:
        for i, label in enumerate(CLASS_LABELS):
            mask = labels == i
            ax.scatter(embed[mask, 0], embed[mask, 1],
                       c=[colors[i]], label=label, s=15, alpha=0.7, edgecolors='none')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1 (arbitrary)', fontsize=11)
        ax.set_ylabel('t-SNE Dim 2 (arbitrary)', fontsize=11)
        ax.legend(markerscale=2, fontsize=9, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f't-SNE Visualization — {args.data}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = os.path.join(result_path, f'tsne_{args.data}_all_p{perplexity}_{now_str}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
# ============================================================
# [TESIS] Modos V-A (no existen en timnet original).
# Todo lo que sigue hasta el final del archivo es aporte de la tesis:
#   - va_train : K-fold con cabezal dual (clasificación + V-A)
#   - va_tsne  : compara baseline vs V-A en t-SNE + circumplex
# ============================================================
elif args.mode == "va_train":
    model.va_train(x_source, y_source)
elif args.mode == "va_tsne":
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import datetime

    print("=" * 60)
    print("Stage 1/2: Training BASELINE model (TIM-Net + WeightLayer)")
    print("=" * 60)
    raw_feats, baseline_feats, labels = model.train_tsne(x_source, y_source)

    print("=" * 60)
    print("Stage 2/2: Training V-A model (TIM-Net + VAFusionLayer)")
    print("=" * 60)
    va_feats, labels_va, va_pred_all = model.train_va_tsne(x_source, y_source)

    assert np.array_equal(labels, labels_va), "Labels mismatch between baseline and V-A runs"

    perplexity = 30
    if perplexity >= labels.shape[0]:
        perplexity = max(5, labels.shape[0] // 3)

    print(f"Running t-SNE (perplexity={perplexity}) on 3 feature sets...")
    tsne_raw = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embed_raw = tsne_raw.fit_transform(raw_feats)

    tsne_baseline = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embed_baseline = tsne_baseline.fit_transform(baseline_feats)

    tsne_va = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embed_va = tsne_va.fit_transform(va_feats)

    num_classes = len(CLASS_LABELS)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, embed, title in [
        (axes[0], embed_raw, "Raw MFCC (T×39 dims, flattened)"),
        (axes[1], embed_baseline, "Baseline TIM-Net — WeightLayer output (39-dim)"),
        (axes[2], embed_va, "V-A Fusion TIM-Net — VAFusionLayer output (39-dim)")
    ]:
        for i, label in enumerate(CLASS_LABELS):
            mask = labels == i
            ax.scatter(embed[mask, 0], embed[mask, 1],
                       c=[colors[i]], label=label, s=15, alpha=0.7, edgecolors='none')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1 (arbitrary)', fontsize=11)
        ax.set_ylabel('t-SNE Dim 2 (arbitrary)', fontsize=11)
        ax.legend(markerscale=2, fontsize=9, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f't-SNE Comparison — {args.data}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = os.path.join(result_path, f'tsne_va_compare_{args.data}_seed{args.random_seed}_p{perplexity}_{now_str}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")

    from va_visualization import plot_va_circumplex_predictions
    circumplex_file = os.path.join(result_path, f'va_circumplex_predictions_{args.data}_seed{args.random_seed}_{now_str}.png')
    plot_va_circumplex_predictions(va_pred_all, labels, CLASS_LABELS, args.data, circumplex_file)
