
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, Model, regularizers, constraints
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, TerminateOnNaN
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ===================== Args =====================
parser = argparse.ArgumentParser(description='RGB 멀티라벨 (MobileNetV2, ASL train + weighted-BCE val)')

# 경로/모델
parser.add_argument('--dataset_path', default='/home/user/chae_project/vgg_data/0805_224/')
parser.add_argument('--checkpoint_path', default='/home/user/chae_project/model_save/')
parser.add_argument('--model_name', default='mobilenet')
parser.add_argument('--model_num', default='6')     # ← mobilenet_6
parser.add_argument('--gpu_num', default="0,1,2,3")

# 학습 하이퍼
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--warmup_epochs', type=int, default=8)
parser.add_argument('--BATCH_SIZE', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--stage2_lr', type=float, default=2e-5)
parser.add_argument('--scale_lr', action='store_true', default=False)
parser.add_argument('--use_adamw', action='store_true', default=True)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--cosine_decay', action='store_true', default=True)
parser.add_argument('--unfreeze_from', type=int, default=-80)

# 조기종료/스케줄
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--patience', type=int, default=15)

# 데이터/샘플링
parser.add_argument('--top_k', type=int, default=2)
parser.add_argument('--max_d21', type=int, default=2400)
parser.add_argument('--d21_multi_ratio', type=float, default=0.35)
parser.add_argument('--os_d1', type=int, default=4)
parser.add_argument('--os_d2', type=int, default=5)
parser.add_argument('--os_d11', type=int, default=4)
parser.add_argument('--os_d19', type=int, default=4)

# 모델 옵션
parser.add_argument('--alpha', type=float, default=1.0)

# ASL 하이퍼
parser.add_argument('--gamma_pos', type=float, default=0.0)
parser.add_argument('--gamma_neg', type=float, default=4.0)
parser.add_argument('--neg_clip',  type=float, default=0.08)
parser.add_argument('--label_smoothing', type=float, default=0.02)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
tf.keras.backend.set_floatx('float32')

# ===================== GPU mem growth =====================
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

# ===================== Const =====================
CLASS_LIST = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19','disease21']
NUM_CLASSES = len(CLASS_LIST)
EPS = 1e-7
TOP_K = args.top_k
THRESHOLD_VAR = None  # strategy.scope에서 초기화

# ===================== Utils =====================
def _sanitize(x, fill=0.0):
    return tf.where(tf.math.is_finite(x), x, tf.cast(fill, x.dtype))

def _sigmoid_probs_from_logits(logits: tf.Tensor):
    logits = _sanitize(logits, 0.0)
    p = tf.math.sigmoid(logits)
    return tf.clip_by_value(_sanitize(p, 0.5), 1e-6, 1.0 - 1.0e-6)

def binarize_with_threshold_and_topk_from_logits(logits: tf.Tensor, thresholds: tf.Tensor, k: int):
    y_prob = _sigmoid_probs_from_logits(logits)
    y_bin = tf.cast(y_prob > thresholds, tf.float32)
    if not k or k <= 0:
        return y_bin
    c = tf.shape(y_prob)[1]
    k_eff = tf.minimum(k, c)
    _, topk_idx = tf.math.top_k(y_prob, k=k_eff, sorted=False)
    topk_mask = tf.reduce_max(tf.one_hot(topk_idx, depth=c, dtype=tf.float32), axis=1)
    return y_bin * topk_mask

def normalize_01(x):
    if x.dtype != np.float32:
        x = x.astype('float32')
    x /= 255.0
    return x

# ===================== Losses =====================
def stable_asl_train_loss(pos_w, neg_w, gamma_pos=0.0, gamma_neg=4.0, neg_clip=0.08, label_smoothing=0.02):
    pos_w = tf.cast(pos_w, tf.float32); neg_w = tf.cast(neg_w, tf.float32)
    def loss_fn(y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_true = tf.clip_by_value(_sanitize(y_true, 0.0), 0.0, 1.0)
        logits = _sanitize(logits, 0.0)

        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
        ce = _sanitize(ce, 0.0)
        p = _sigmoid_probs_from_logits(logits)

        w_pos = tf.pow(tf.clip_by_value(1.0 - p, 0.0, 1.0), gamma_pos)
        w_neg = tf.pow(tf.clip_by_value(p, 0.0, 1.0), gamma_neg)
        if neg_clip and neg_clip > 0.0:
            w_neg = w_neg * tf.clip_by_value(1.0 - p + neg_clip, 0.0, 1.0)

        w = y_true * pos_w * tf.stop_gradient(w_pos) + (1.0 - y_true) * neg_w * tf.stop_gradient(w_neg)
        w = _sanitize(w, 1.0)
        loss = _sanitize(w * ce, 0.0)
        return tf.reduce_mean(loss)
    return loss_fn

def weighted_bce_val_loss(pos_w, neg_w, label_smoothing=0.0):
    pos_w = tf.cast(pos_w, tf.float32); neg_w = tf.cast(neg_w, tf.float32)
    def loss_fn(y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_true = tf.clip_by_value(_sanitize(y_true, 0.0), 0.0, 1.0)
        logits = _sanitize(logits, 0.0)

        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
        w  = y_true * pos_w + (1.0 - y_true) * neg_w

        num = tf.reduce_sum(w * ce, axis=0)
        den = tf.reduce_sum(w,     axis=0) + 1e-6
        per_class = num / den
        return tf.reduce_mean(per_class)
    return loss_fn

# ===================== Metrics =====================
def accuracy_logits(y_true, logits):
    y_true = tf.cast(tf.clip_by_value(_sanitize(y_true, 0.0), 0.0, 1.0), tf.float32)
    p = _sigmoid_probs_from_logits(logits)
    y_pred = tf.cast(p > 0.5, tf.float32)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total = tf.cast(tf.size(y_true), tf.float32)
    return correct / (total + EPS)

def precision(y_true, logits):
    y_bin = binarize_with_threshold_and_topk_from_logits(logits, THRESHOLD_VAR, TOP_K)
    tp = tf.reduce_sum(y_true * y_bin); fp = tf.reduce_sum((1.0 - y_true) * y_bin)
    return tp / (tp + fp + EPS)

def recall(y_true, logits):
    y_bin = binarize_with_threshold_and_topk_from_logits(logits, THRESHOLD_VAR, TOP_K)
    tp = tf.reduce_sum(y_true * y_bin); fn = tf.reduce_sum(y_true * (1.0 - y_bin))
    return tp / (tp + fn + EPS)

def micro_f1(y_true, logits):
    y_bin = binarize_with_threshold_and_topk_from_logits(logits, THRESHOLD_VAR, TOP_K)
    tp = tf.reduce_sum(y_true * y_bin)
    fp = tf.reduce_sum((1.0 - y_true) * y_bin)
    fn = tf.reduce_sum(y_true * (1.0 - y_bin))
    return 2.0 * tp / (2.0 * tp + fp + fn + EPS)

# ===================== Model build =====================
def build_model(class_bias=None, alpha=1.0):
    backbone = MobileNetV2(include_top=False, input_shape=(224, 224, 3),
                           weights='imagenet', alpha=alpha)
    x_in = layers.Input(shape=(224, 224, 3))
    z = preprocess_input(x_in * 255.0)
    z = backbone(z, training=False)
    z = layers.GlobalAveragePooling2D()(z)
    z = layers.Dense(1024, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=constraints.MaxNorm(3))(z)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=constraints.MaxNorm(3))(z)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=constraints.MaxNorm(3))(z)
    bias_init = tf.keras.initializers.Constant(class_bias) if class_bias is not None else 'zeros'
    out = layers.Dense(NUM_CLASSES, activation=None, bias_initializer=bias_init,
                       kernel_constraint=constraints.MaxNorm(3))(z)
    base = Model(inputs=x_in, outputs=out, name="mobilenet_asl")
    for l in backbone.layers: l.trainable = False
    base._backbone_layers = backbone.layers
    return base

def set_backbone_trainable(model: tf.keras.Model, from_idx: int):
    layers_ = getattr(model, "_backbone_layers", None)
    if layers_ is None:
        try: layers_ = model.get_layer('MobilenetV2').layers
        except: raise RuntimeError("Backbone layers not found.")
    for l in layers_: l.trainable = False
    start = len(layers_) + from_idx if from_idx < 0 else from_idx
    start = max(0, min(start, len(layers_)))
    for l in layers_[start:]: l.trainable = True

# ===================== Subclassed model =====================
class ASLModel(tf.keras.Model):
    def __init__(self, base_model, loss_val_fn, clip_global_norm=1.0):
        super().__init__()
        self.base = base_model
        self.loss_val_fn = loss_val_fn
        self.clip_global_norm = clip_global_norm

    def call(self, x, training=False):
        return self.base(x, training=training)  # logits

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)       # logits
            loss = self.compiled_loss(y, y_pred)  # ASL
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip_global_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)          # logits
        val_loss = self.loss_val_fn(y, y_pred)    # 클래스가중 BCE
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = val_loss
        return results

# ===================== Aug/Dataset =====================
def make_augmenter(strength='normal'):
    if strength == 'strong':
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.35),
            layers.RandomZoom(0.35),
            layers.RandomTranslation(0.25, 0.25),
            layers.RandomContrast(0.25),
            layers.Resizing(224, 224),
        ])
    else:
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),
            layers.Resizing(224, 224),
        ])

def make_dataset(x, y, batch_size, training=True, strong=False, shuffle_buffer=2048):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), shuffle_buffer), reshuffle_each_iteration=True)

    def norm_map(a, b):
        a = tf.cast(a, tf.float32) / 255.0
        b = tf.cast(b, tf.float32)
        a = _sanitize(a, 0.0)
        b = tf.clip_by_value(_sanitize(b, 0.0), 0.0, 1.0)
        return a, b

    if training:
        aug = make_augmenter('strong' if strong else 'normal')
        ds = ds.map(norm_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        resize = tf.keras.Sequential([layers.Resizing(224, 224)])
        ds = ds.map(norm_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda a, b: (resize(a, training=False), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ===================== Sampling =====================
def build_indices_for_training(y, max_d21, oversample_factors, d21_multi_ratio=0.35):
    y = np.asarray(y)
    d21_only  = np.where((y[:,7]==1) & (y.sum(axis=1)==1))[0]
    d21_multi = np.where((y[:,7]==1) & (y.sum(axis=1)>1))[0]
    np.random.shuffle(d21_multi)
    d21_multi = d21_multi[: int(len(d21_only) * d21_multi_ratio)] if len(d21_only)>0 else d21_multi[:0]
    d21_idx = np.concatenate([d21_only, d21_multi]) if len(d21_multi)>0 else d21_only
    np.random.shuffle(d21_idx)
    d21_idx = d21_idx[:max_d21]

    non21_idx = np.where(y[:,7]==0)[0]
    base_idx = np.concatenate([non21_idx, d21_idx])

    extra = []
    for cls, factor in oversample_factors.items():
        if factor <= 1: continue
        pos = np.where(y[:,cls]==1)[0]
        if len(pos)==0: continue
        rep = np.random.choice(pos, size=len(pos)*(factor-1), replace=True)
        extra.append(rep)
    if extra:
        base_idx = np.concatenate([base_idx] + extra)
    np.random.shuffle(base_idx)
    return base_idx

# ===================== Threshold tuner =====================
class ClasswiseThresholdTuner(Callback):
    def __init__(self, x_val, y_val, every=3, grid=np.arange(0.30, 0.96, 0.01), iters=1, save_dir=None, top_k=None):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val.astype(np.int32)
        self.every = max(1, every)
        self.grid = grid; self.iters = max(1, iters)
        self.save_dir = save_dir; self.top_k = top_k

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0: return
        logits = self.model.predict(self.x_val, batch_size=32, verbose=0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        prob = np.clip(prob, 1e-6, 1.0 - 1.0e-6)

        thr = THRESHOLD_VAR.numpy().copy()
        def make_pred(p, thr_vec):
            y_pred = (p > thr_vec[None, :]).astype(np.int32)
            if self.top_k and self.top_k > 0:
                for i in range(len(y_pred)):
                    pos = np.where(y_pred[i] == 1)[0]
                    if len(pos) > self.top_k:
                        keep = pos[np.argsort(-p[i, pos])[:self.top_k]]
                        y_pred[i] = 0; y_pred[i, keep] = 1
            return y_pred

        best_f1 = f1_score(self.y_val, make_pred(prob, thr), average='micro', zero_division=0)
        for _ in range(self.iters):
            for c in range(NUM_CLASSES):
                best_t = thr[c]
                for t in self.grid:
                    thr[c] = t
                    f1 = f1_score(self.y_val, make_pred(prob, thr), average='micro', zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                thr[c] = best_t

        THRESHOLD_VAR.assign(thr.astype(np.float32))
        print(f"\n[ClasswiseTuner] epoch {epoch+1}: val_micro_f1={best_f1:.4f}, thr={np.round(thr,2)}")
        if self.save_dir:
            np.save(os.path.join(self.save_dir, f"classwise_thr_epoch{epoch+1:03d}.npy"), thr.astype(np.float32))

# ===================== Temp scaling + Final threshold search =====================
def fit_temperature(logits, y, grid=np.linspace(0.7, 1.6, 19)):
    T = np.ones(logits.shape[1], dtype=np.float32)
    for c in range(logits.shape[1]):
        best_t, best_loss = 1.0, 1e9
        z = logits[:, c]; yc = y[:, c]
        for t in grid:
            p = 1.0 / (1.0 + np.exp(-z / t))
            eps = 1e-6
            loss = -np.mean(yc*np.log(p+eps) + (1-yc)*np.log(1-p+eps))
            if loss < best_loss:
                best_loss, best_t = loss, t
        T[c] = best_t
    return T

def search_best_threshold(model, x_val, y_val, temp_scale=None):
    logits = model.predict(x_val, batch_size=32, verbose=0)
    if temp_scale is None:
        temp_scale = np.ones((NUM_CLASSES,), np.float32)
    prob = 1.0 / (1.0 + np.exp(-(logits / temp_scale[None, :])))
    prob = np.clip(prob, 1e-6, 1.0 - 1.0e-6)
    thr = THRESHOLD_VAR.numpy().copy()
    grid = np.arange(0.30, 0.96, 0.01)

    def score(thr_vec):
        y_pred = (prob > thr_vec[None,:]).astype(int)
        if TOP_K and TOP_K > 0:
            for i in range(len(y_pred)):
                pos = np.where(y_pred[i]==1)[0]
                if len(pos) > TOP_K:
                    keep = pos[np.argsort(-prob[i, pos])[:TOP_K]]
                    y_pred[i] = 0; y_pred[i, keep] = 1
        return f1_score(y_val, y_pred, average='micro', zero_division=0)

    best_f1 = score(thr)
    for _ in range(2):
        for c in range(NUM_CLASSES):
            best_t = thr[c]
            for t in grid:
                thr[c] = t
                f1 = score(thr)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thr[c] = best_t
    print(f"[INFO] Best thresholds(final)={np.round(thr,2)}, F1={best_f1:.4f}")
    return thr.astype(np.float32)

# ===================== Plot =====================
def plot_loss(history, args):
    plt.figure()
    plt.plot(history.history.get('loss', []), '+-', label='train_loss')
    plt.plot(history.history.get('val_loss', []), 'x-', label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    path = os.path.join(args.checkpoint_path, f"{args.model_name}_{args.model_num}_loss.png")
    plt.savefig(path); plt.close()
    print(f"[INFO] Loss 그래프 저장 완료: {path}")

# ===================== Main =====================
def main(args):
    global THRESHOLD_VAR, TOP_K

    # 데이터 로드
    train_df = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(args.dataset_path, 'valid.csv'))
    assert list(train_df.columns[1:]) == CLASS_LIST
    assert list(valid_df.columns[1:]) == CLASS_LIST

    x_train = np.load(os.path.join(args.dataset_path, 'train.npy'))
    x_valid = np.load(os.path.join(args.dataset_path, 'valid.npy'))
    y_train = train_df.iloc[:, 1:].values.astype('float32')
    y_valid = valid_df.iloc[:, 1:].values.astype('float32')

    y_train = np.clip(np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    y_valid = np.clip(np.nan_to_num(y_valid, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    # 샘플링
    oversample_factors = {0: args.os_d1, 1: args.os_d2, 4: args.os_d11, 6: args.os_d19}
    train_idx = build_indices_for_training(
        y_train, max_d21=args.max_d21,
        oversample_factors=oversample_factors,
        d21_multi_ratio=args.d21_multi_ratio
    )
    x_train_sel, y_train_sel = x_train[train_idx], y_train[train_idx]
    print(f"[INFO] train selected: {len(train_idx)} / valid: {len(x_valid)}")

    # 클래스 가중치
    counts_sel = y_train_sel.sum(axis=0).astype(np.float32)
    max_c = counts_sel.max() if counts_sel.max() > 0 else 1.0
    pos_w = np.clip(max_c / np.maximum(counts_sel, 1.0), 0.5, 5.0)
    neg_w = np.clip(1.0 + (counts_sel / max_c), 1.0, 2.0).astype(np.float32)
    # 13/21 FP 억제
    pos_w[[5, 7]] = np.minimum(pos_w[[5, 7]], 1.2)
    neg_boost = np.ones_like(neg_w, np.float32); neg_boost[[3,5,7]] = [1.15, 1.55, 1.75]
    neg_w = np.clip(neg_w * neg_boost, 1.0, 3.0)

    print("[INFO] class counts (selected):", dict(zip(CLASS_LIST, counts_sel.astype(int))))
    print("[INFO] POS_W (from selected):", dict(zip(CLASS_LIST, np.round(pos_w,3))))
    print("[INFO] NEG_W (from selected):", dict(zip(CLASS_LIST, np.round(neg_w,3))))

    pos_w_tf = tf.constant(pos_w, tf.float32)
    neg_w_tf = tf.constant(neg_w, tf.float32)

    # 초기 바이어스
    prev = np.clip(y_train_sel.mean(axis=0), 1e-4, 1-1e-4)
    class_bias = np.log(prev / (1.0 - prev)).astype('float32')

    # 멀티 GPU
    strategy = tf.distribute.MirroredStrategy()
    replicas = strategy.num_replicas_in_sync
    print(f"[INFO] Using MirroredStrategy with {replicas} GPUs")
    global_batch = int(np.ceil(args.BATCH_SIZE / replicas) * replicas)
    if global_batch != args.BATCH_SIZE:
        print(f"[WARN] BATCH_SIZE {args.BATCH_SIZE} → {global_batch}")
    base_lr = args.learning_rate * replicas if args.scale_lr else args.learning_rate

    # 데이터 파이프라인
    ds_train = make_dataset(x_train_sel, y_train_sel, batch_size=global_batch, training=True, strong=False)
    ds_valid = make_dataset(x_valid, y_valid, batch_size=global_batch, training=False)

    # 체크포인트
    model_dir = os.path.join(args.checkpoint_path, f"{args.model_name}_{args.model_num}")
    os.makedirs(model_dir, exist_ok=True)
    best_prefix = os.path.join(model_dir, 'best_model')

    # ===== scope 내부에서 생성/컴파일/이어학습 로드 =====
    with strategy.scope():
        THRESHOLD_VAR = tf.Variable([0.35, 0.35, 0.40, 0.60, 0.35, 0.65, 0.35, 0.70],
                                    dtype=tf.float32, trainable=False)

        base = build_model(class_bias=class_bias, alpha=args.alpha)

        # 손실 함수
        loss_train_fn = stable_asl_train_loss(
            pos_w_tf, neg_w_tf,
            gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg,
            neg_clip=args.neg_clip, label_smoothing=args.label_smoothing
        )
        loss_val_fn   = weighted_bce_val_loss(pos_w_tf, neg_w_tf, label_smoothing=0.0)

        # 옵티마이저
        total_steps = int(np.ceil(len(x_train_sel) / float(global_batch))) * (args.epoch)
        if args.cosine_decay:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=base_lr, first_decay_steps=max(total_steps//8, 1000), t_mul=2.0, m_mul=0.5
            )
        else:
            lr_schedule = base_lr

        if args.use_adamw:
            try:
                opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
            except:
                opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
        else:
            opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

        # 모델 생성
        model = ASLModel(base_model=base, loss_val_fn=loss_val_fn, clip_global_norm=1.0)
        # 더미 콜로 빌드(이어학습 로딩 안정화)
        _ = model(tf.zeros([1, 224, 224, 3]), training=False)

        # 이어학습: 반드시 ASLModel에 로드!
        resume_in_place = os.path.exists(best_prefix + ".index")
        if resume_in_place:
            model.load_weights(best_prefix)
            print(f"[INFO] Resume detected → load {best_prefix}")
            # 이어학습은 고정 LR로 짧게
            if args.use_adamw:
                try:
                    opt = tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=args.weight_decay)
                except:
                    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=5e-5, weight_decay=args.weight_decay)
            else:
                opt = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
            args.warmup_epochs = min(args.warmup_epochs, 3)
            args.stage2_lr = 2e-5
            print(f"[INFO] Resume settings → LR=5e-05, warmup_epochs={args.warmup_epochs}, stage2_lr={args.stage2_lr}")

        model.compile(
            optimizer=opt,
            loss=loss_train_fn,
            metrics=[accuracy_logits, precision, recall, micro_f1],
        )

    callbacks1 = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_micro_f1', patience=args.patience, min_delta=args.delta, mode='max', verbose=1),
        ModelCheckpoint(os.path.join(model_dir, 'best_model'),
                        monitor='val_micro_f1', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_micro_f1', mode='max', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ReduceLROnPlateau(monitor='val_loss',     mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ClasswiseThresholdTuner(x_val=normalize_01(x_valid), y_val=y_valid, every=3,
                                grid=np.arange(0.30,0.96,0.01), iters=1, save_dir=model_dir, top_k=TOP_K),
    ]

    # Stage 1
    hist1 = model.fit(ds_train,
                      validation_data=ds_valid,
                      epochs=args.warmup_epochs,
                      callbacks=callbacks1,
                      verbose=1)

    # ===== Stage 2: 언프리즈 + 낮은 LR =====
    with strategy.scope():
        set_backbone_trainable(model.base, args.unfreeze_from)
        if args.use_adamw:
            try:
                opt2 = tf.keras.optimizers.AdamW(learning_rate=args.stage2_lr, weight_decay=args.weight_decay)
            except:
                opt2 = tf.keras.optimizers.experimental.AdamW(learning_rate=args.stage2_lr, weight_decay=args.weight_decay)
        else:
            opt2 = tf.keras.optimizers.legacy.Adam(learning_rate=args.stage2_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

        model.compile(
            optimizer=opt2,
            loss=loss_train_fn,
            metrics=[accuracy_logits, precision, recall, micro_f1],
        )

    callbacks2 = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_micro_f1', patience=args.patience, min_delta=args.delta, mode='max', verbose=1),
        ModelCheckpoint(os.path.join(model_dir, 'best_model'),
                        monitor='val_micro_f1', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_micro_f1', mode='max', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ReduceLROnPlateau(monitor='val_loss',     mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ClasswiseThresholdTuner(x_val=normalize_01(x_valid), y_val=y_valid, every=3,
                                grid=np.arange(0.30,0.96,0.01), iters=1, save_dir=model_dir, top_k=TOP_K),
    ]

    hist2 = model.fit(ds_train,
                      validation_data=ds_valid,
                      epochs=args.epoch,
                      initial_epoch=args.warmup_epochs,
                      callbacks=callbacks2,
                      verbose=1)

    # ----- Temperature scaling -----
    x_val_n = normalize_01(x_valid)
    logits_val = model.predict(x_val_n, batch_size=32, verbose=0)
    T = fit_temperature(logits_val, y_valid)
    np.save(os.path.join(model_dir, "temp_scale.npy"), T)
    print(f"[INFO] Temperature saved: {np.round(T,2)}")

    # ----- 최종 threshold 저장 -----
    best_thr = search_best_threshold(model, x_val_n, y_valid, temp_scale=T)
    np.save(os.path.join(model_dir, "best_threshold.npy"), best_thr)
    print(f"[INFO] Best threshold saved (per-class): {np.round(best_thr,2)}")

    # 모델 저장
    model.save_weights(os.path.join(model_dir, 'saved_weights'))
    print("[INFO] 모델 저장 완료 (weights only).")

    # 로그 병합 후 그래프 저장
    history = {}
    for h in (hist1.history, hist2.history):
        for k,v in h.items(): history.setdefault(k, []).extend(v)
    class H: pass
    Hobj = H(); Hobj.history = history
    plot_loss(Hobj, args)

if __name__ == '__main__':
    main(args)
