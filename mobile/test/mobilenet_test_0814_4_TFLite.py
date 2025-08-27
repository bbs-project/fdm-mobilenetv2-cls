# test_mobilenet_asl_tflite.py
import os
import argparse
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, multilabel_confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ==================== Args ====================
parser = argparse.ArgumentParser(description='MobileNetV2-ASL TFLite 테스트')
parser.add_argument('--model_num', type=str, default='8')
parser.add_argument('--checkpoint_path', type=str, default='/home/user/chae_project/model_save/')
parser.add_argument('--dataset_path', type=str, default='/home/user/chae_project/vgg_data/0805_224/')
parser.add_argument('--tflite_path', type=str, default=None, help='직접 지정할 TFLite 경로')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--samples', type=int, default=10)
parser.add_argument('--num_threads', type=int, default=4)

# === 임계값/후처리 옵션 ===
parser.add_argument('--d21_min_thr', type=float, default=None)     # d21 임계값 하한 (예: 0.62)
parser.add_argument('--top_k_override', type=int, default=None)    # top-k 강제
parser.add_argument('--use_temp', action='store_true', default=True)   # temp_scale.npy 사용
parser.add_argument('--try_topk1', action='store_true', default=False) # top_k=1 비교

# === d21 배타 옵션 ===d
parser.add_argument('--d21_exclusive', action='store_true', default=False,
                    help='d21이 확실하면 d21만 1로, 나머지 0')
parser.add_argument('--d21_margin', type=float, default=0.06,
                    help='2위보다 이만큼 높을 때만 d21 배타 적용')

args = parser.parse_args()
tf.keras.backend.set_floatx('float32')

# ==================== Const ====================
CLASS_LIST = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19','disease21']
NUM_CLASSES = len(CLASS_LIST)
EPS = 1e-7

# ==================== Utils ====================
def get_model_dir():
    return os.path.join(args.checkpoint_path, f'mobilenet_{args.model_num}')

def get_threshold_path():
    return os.path.join(get_model_dir(), 'best_threshold.npy')

def get_temp_scale_path():
    return os.path.join(get_model_dir(), 'temp_scale.npy')

def get_default_tflite_path():
    return os.path.join(get_model_dir(), f"mobilenet_{args.model_num}_float32.tflite")

def normalize_01(x):
    if x.dtype != np.float32:
        x = x.astype('float32')
    x /= 255.0
    return x

def binarize_with_threshold_and_topk(y_prob, thresholds, top_k):
    if np.isscalar(thresholds):
        thr = np.full((y_prob.shape[1],), float(thresholds), dtype=np.float32)
    else:
        thr = thresholds.astype(np.float32)
    y_bin = (y_prob > thr[None, :]).astype(np.int32)
    if not top_k or top_k <= 0:
        return y_bin
    out = np.zeros_like(y_bin)
    for i in range(y_prob.shape[0]):
        idx = np.argsort(-y_prob[i])[:top_k]
        out[i, idx] = 1
    return (y_bin & out).astype(np.int32)

def ensure_per_class_thresholds(thresholds, num_classes):
    if np.isscalar(thresholds):
        return np.full((num_classes,), float(thresholds), dtype=np.float32)
    return thresholds.astype(np.float32)

def load_thresholds():
    thr_path = get_threshold_path()
    if os.path.exists(thr_path):
        arr = np.load(thr_path)
        if arr.ndim == 0:
            thr = float(arr)
            print(f"[INFO] 임계값 로드(스칼라): {thr:.2f}")
            return thr
        elif arr.ndim == 1 and arr.shape[0] == NUM_CLASSES:
            print(f"[INFO] 임계값 로드(클래스별): {np.round(arr,2)}")
            return arr.astype(np.float32)
    print("[INFO] 임계값 파일 없음 → 기본 0.5 사용")
    return 0.5

def load_temp_scale():
    p = get_temp_scale_path()
    if os.path.exists(p):
        T = np.load(p).astype(np.float32)
        if T.ndim == 1 and T.shape[0] == NUM_CLASSES:
            print(f"[INFO] temp_scale 로드: {np.round(T,2)}")
            return T
    print("[INFO] temp_scale 없음 → 미사용")
    return None

def apply_temperature_to_probs(y_prob, temp_vec):
    """
    TFLite는 확률(sigmoid)로 출력하므로, temp를 logits에 적용한 효과를
    logit(p)/T -> sigmoid 로 근사 반영.
    """
    temp_vec = np.clip(temp_vec.astype(np.float32), 1e-3, 10.0)[None, :]
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    scaled = logit / temp_vec
    return 1.0 / (1.0 + np.exp(-scaled))

# --- d21 배타 후처리 ---
def apply_d21_exclusive(y_prob, y_bin, thr_vec, margin=0.06):
    second_best = np.partition(y_prob, -2, axis=1)[:, -2]
    cond = (y_prob[:, 7] >= thr_vec[7]) & ((y_prob[:, 7] - second_best) >= float(margin))
    if np.any(cond):
        y_bin[cond, :] = 0
        y_bin[cond, 7] = 1
    return y_bin

# ==================== Data ====================
def load_test_data():
    csv_path = os.path.join(args.dataset_path, 'test.csv')
    npy_path = os.path.join(args.dataset_path, 'test.npy')
    df = pd.read_csv(csv_path)
    x = np.load(npy_path)
    x = normalize_01(x)   # [0..1]
    y = df[CLASS_LIST].values.astype(np.int32)
    filenames = df['file_name'].values
    return x, y, filenames

# ==================== TFLite ====================
def load_tflite_interpreter(tflite_path, num_threads=4):
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite 파일 없음: {tflite_path}")
    delegates = []
    try:
        # XNNPACK delegate (CPU 가속)
        delegates.append(tf.lite.experimental.load_delegate('libtensorflowlite_flex_delegate.so'))
    except Exception:
        # Flex delegate가 없을 수 있음 → 무시하고 기본 인터프리터 사용
        delegates = []
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter

def tflite_predict(interpreter, x, batch_size):
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    in_index = in_details['index']
    out_index = out_details['index']
    in_dtype = in_details['dtype']
    out_dtype = out_details['dtype']
    in_scale, in_zero = in_details.get('quantization', (0.0, 0))
    out_scale, out_zero = out_details.get('quantization', (0.0, 0))

    probs_list = []
    n = len(x)
    start = time.time()
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        batch = x[i:j].astype(np.float32)  # [0..1] float

        # 배치 크기에 맞게 리사이즈
        interpreter.resize_tensor_input(in_index, [j - i, 224, 224, 3])
        interpreter.allocate_tensors()

        if in_dtype == np.float32:
            interpreter.set_tensor(in_index, batch)
        elif in_dtype == np.uint8 or in_dtype == np.int8:
            # 양자 입력 지원 (일반적으로 이 스크립트는 float32 I/O일 것)
            q = (batch / (in_scale if in_scale > 0 else 1.0) + in_zero).round()
            q = np.clip(q, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
            interpreter.set_tensor(in_index, q)
        else:
            raise TypeError(f"지원되지 않는 입력 dtype: {in_dtype}")

        interpreter.invoke()
        out = interpreter.get_tensor(out_index)

        if out_dtype == np.float32:
            probs = out.astype(np.float32)
        elif out_dtype == np.uint8 or out_dtype == np.int8:
            probs = (out.astype(np.float32) - out_zero) * (out_scale if out_scale > 0 else 1.0)
        else:
            raise TypeError(f"지원되지 않는 출력 dtype: {out_dtype}")

        probs_list.append(probs)
        i = j

    y_prob = np.concatenate(probs_list, axis=0)
    elapsed = time.time() - start
    return y_prob, elapsed

# ==================== Eval ====================
def evaluate_with_tflite(tflite_path, thresholds, x, y_true, filenames,
                         batch_size=128, temp_scale=None, top_k=None,
                         print_samples=True, tag=""):
    interpreter = load_tflite_interpreter(tflite_path, num_threads=args.num_threads)
    y_prob, elapsed = tflite_predict(interpreter, x, batch_size)

    # (선택) 온도 보정 – 확률→logit→T 적용→확률
    if temp_scale is not None:
        y_prob = apply_temperature_to_probs(y_prob, temp_scale)

    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

    tk = args.top_k if top_k is None else top_k
    thr_vec = ensure_per_class_thresholds(thresholds, NUM_CLASSES)
    y_bin = binarize_with_threshold_and_topk(y_prob, thr_vec, tk)

    # d21 배타 규칙
    if args.d21_exclusive:
        y_bin = apply_d21_exclusive(y_prob, y_bin, thr_vec, margin=args.d21_margin)

    excl = 'on' if args.d21_exclusive else 'off'
    tag_str = f"[{tag}, exclusive={excl}]" if tag else f"[exclusive={excl}]"
    print(f"\n[INFO] 예측 완료 (총 소요 시간: {elapsed:.2f}초) {tag_str}")
    print(f"예측 확률 평균: {np.round(np.mean(y_prob, axis=0), 4)}")
    print(f"예측 확률 분포: {np.min(y_prob):.4f} ~ {np.max(y_prob):.4f}")
    print(f"클래스별 1 예측 비율: {np.round(np.mean(y_bin, axis=0), 4)}\n")

    d21_mean = float(np.mean(y_prob[:, 7]))
    d21_posrate = float(np.mean(y_bin[:, 7]))
    d21_above_thr = float(np.mean(y_prob[:, 7] > thr_vec[7]))
    print(f"[d21] mean_prob={d21_mean:.4f} / thr={thr_vec[7]:.2f} / "
          f"rate_above_thr={d21_above_thr:.4f} / predicted_pos={d21_posrate:.4f}")

    # Metrics
    f1_micro = f1_score(y_true, y_bin, average='micro', zero_division=0)
    f1_round = f1_score(y_true, np.round(y_prob), average='micro', zero_division=0)
    print(f"F1 Score (micro, binarized):  {f1_micro * 100:.2f}%")
    print(f"F1 Score (legacy, round):     {f1_round * 100:.2f}%")

    prec = precision_score(y_true, y_bin, average='micro', zero_division=0)
    rec_sklearn = recall_score(y_true, y_bin, average='micro', zero_division=0)
    cm = multilabel_confusion_matrix(y_true, y_bin)
    TP_sum = sum(cm[i][1,1] for i in range(NUM_CLASSES))
    FN_sum = sum(cm[i][1,0] for i in range(NUM_CLASSES))
    rec_manual = TP_sum / (TP_sum + FN_sum + 1e-8)

    print(f"Precision (micro):            {prec * 100:.2f}%")
    print(f"Recall (sklearn):             {rec_sklearn * 100:.2f}%")
    print(f"Recall (manual):              {rec_manual * 100:.2f}%")

    spec_list = []
    for i in range(NUM_CLASSES):
        tn = cm[i][0,0]
        fp = cm[i][0,1]
        spec = tn / (tn + fp + 1e-8)
        spec_list.append(spec)
    print(f"Specificity (avg):            {np.mean(spec_list) * 100:.2f}%")

    auprcs = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    print(f"AUPRC (avg):                  {np.mean(auprcs) * 100:.2f}%")

    print(f"\nInference Time/Image:         {elapsed / len(x):.4f} sec\n")

    tflite_size_mb = os.path.getsize(tflite_path) / (1024*1024)
    print(f"Model Size (tflite):          {tflite_size_mb:.2f} MB\n")

    if print_samples:
        print(f"========== Sample Predictions ==========")
        sample_idxs = random.sample(range(len(y_bin)), min(args.samples, len(y_bin)))
        for idx in sample_idxs:
            gt = [CLASS_LIST[i] for i, v in enumerate(y_true[idx]) if v == 1]
            pred = [CLASS_LIST[i] for i, v in enumerate(y_bin[idx]) if v == 1]
            print(f"[{idx:>5}] File: {filenames[idx]}")
            print(f"     GT      : {gt}")
            print(f"     Predict : {pred}\n")

# ==================== Run ====================
if __name__ == '__main__':
    # TFLite 경로 결정
    tflite_path = args.tflite_path or get_default_tflite_path()
    print(f"[INFO] TFLite: {tflite_path}")

    # Threshold / Temp
    thresholds = load_thresholds()
    thresholds = ensure_per_class_thresholds(thresholds, NUM_CLASSES)

    if args.d21_min_thr is not None:
        old = thresholds[7]
        thresholds[7] = max(thresholds[7], float(args.d21_min_thr))
        if thresholds[7] != old:
            print(f"[INFO] d21 thr 상승: {old:.2f} -> {thresholds[7]:.2f}")

    if args.top_k_override is not None:
        print(f"[INFO] top_k override: {args.top_k} -> {args.top_k_override}")
        args.top_k = int(args.top_k_override)

    temp_scale = load_temp_scale() if args.use_temp else None

    # 데이터
    x_test, y_test, filenames = load_test_data()

    # 평가
    evaluate_with_tflite(
        tflite_path, thresholds, x_test, y_test, filenames,
        batch_size=args.batch_size, temp_scale=temp_scale, top_k=args.top_k,
        tag=f"top_k={args.top_k}, temp={'on' if temp_scale is not None else 'off'}"
    )

    if args.try_topk1 and (args.top_k != 1):
        evaluate_with_tflite(
            tflite_path, thresholds, x_test, y_test, filenames,
            batch_size=args.batch_size, temp_scale=temp_scale, top_k=1,
            print_samples=False, tag="top_k=1"
        )
