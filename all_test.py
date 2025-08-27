import os, time, glob, random
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, multilabel_confusion_matrix,
    precision_recall_fscore_support
)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mbv2_pre

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===================== 경로/설정 =====================
DATASET_PATH = "/home/user/chae_project/vgg_data/"
TEST_NPY = os.path.join(DATASET_PATH, "test.npy")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")

# 장당 처리시간 측정용 단일 원본 이미지
ROOT_DIR  = "/data/301.FlounderDiseaseData-2024.03/01-1.정식개방데이터/Training/01.원천데이터/rgb"
FILENAME  = "F02_U01_O2177_D2022-09-15_L365_W0585_S3_R05_B02_I00061319.JPG"

# 모델 경로
TFLITE_VGG = "/home/user/fdm/lite-models/tflite/fdm_vgg16.tflite"         # (입력 112 가정)
KERAS_VGG  = "/home/user/fdm/rgb-classify-vgg16/saved_model/my_model"     # 폴더 안 .h5 또는 SavedModel
MBV2_DIR   = "/home/user/chae_project/model_save/mobilenet_8"             # TFLite + (만든 .h5 / saved_model_export)

# 클래스
FULL_CLASSES   = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19','disease21']  # 8 (MBV2)
LEGACY_CLASSES = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19']              # 7 (VGG: disease21 제외)

# === 임계값(단일) ===
THRESHOLD = 0.8
MAIN_SAMPLE_THR = THRESHOLD

# ===== 전처리 내장 여부 플래그 =====
# TFLite: True면 0..1 그대로, False면 *255 적용
VGG_TFLITE_HAS_PRE  = True
MBV2_TFLITE_HAS_PRE = True
# Keras: True면 0..1 그대로(모델 안 전처리 포함), False면 *255 + preprocess_input
VGG_KERAS_HAS_PRE   = True
MBV2_KERAS_HAS_PRE  = True

# 실행/타이밍 파라미터
KERAS_BATCH = 64
TFLITE_THREADS = 4
TIMING_WARMUP = 5
TIMING_REPEATS = 15
INCLUDE_RESIZE_IN_TIMING = True  # 리사이즈/전처리 포함해서 시간 측정

# ===================== 유틸 =====================
def load_image_raw(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    return img.numpy()

def load_test(test_npy, test_csv):
    if not os.path.exists(test_npy): raise FileNotFoundError(test_npy)
    if not os.path.exists(test_csv): raise FileNotFoundError(test_csv)
    X = np.load(test_npy, mmap_mode="r").astype(np.float32)
    if X.max() > 1.0: X /= 255.0

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    fname_col = 'filename' if 'filename' in df.columns else df.columns[0]
    files = df[fname_col].astype(str).tolist()

    for c in FULL_CLASSES:
        if c not in df.columns:
            raise ValueError(f"test.csv에 '{c}' 컬럼이 필요합니다.")
    Y_full = df[FULL_CLASSES].values.astype(int)

    N = min(len(X), len(files), len(Y_full))
    if N != len(X) or N != len(files):
        print(f"[WARN] Trim to common length: images={len(X)} csv={len(files)} → use {N}")
        X = X[:N]; files = files[:N]; Y_full = Y_full[:N]

    print(f"[INFO] Loaded test set: {N} images")
    return X, files, Y_full

def bytes_to_mb(n_bytes):
    return n_bytes / (1024.0 * 1024.0)

def path_size_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try: total += os.path.getsize(fp)
            except: pass
    return total

# ===================== per-class 지표 =====================
def per_class_specificity(Y_true, Y_bin):
    mcm = multilabel_confusion_matrix(Y_true, Y_bin)
    specs = []
    for k in range(mcm.shape[0]):
        tn, fp = mcm[k][0,0], mcm[k][0,1]
        specs.append(tn / (tn + fp + 1e-12))
    return np.array(specs)

def per_class_report(Y_true, Y_prob, Y_bin, class_list):
    p, r, f1, sup = precision_recall_fscore_support(
        Y_true, Y_bin, average=None, zero_division=0
    )
    auprc = []
    for c in range(Y_true.shape[1]):
        try:
            auprc.append(average_precision_score(Y_true[:, c], Y_prob[:, c]))
        except Exception:
            auprc.append(np.nan)
    auprc = np.array(auprc)
    spec = per_class_specificity(Y_true, Y_bin)
    df = pd.DataFrame({
        "class": class_list,
        "support": sup.astype(int),
        "precision(%)": (p * 100).round(2),
        "recall(%)":    (r * 100).round(2),
        "specificity(%)": (spec * 100).round(2),
        "auprc(%)":     (auprc * 100).round(2),
        "f1(%)":        (f1 * 100).round(2),
    })
    return df

# ===================== 입력 크기 추출 =====================
def _get_hw_from_tflite(interp):
    det = interp.get_input_details()[0]
    shape = det.get('shape_signature', det['shape'])
    h = int(shape[1]) if shape[1] > 0 else 224
    w = int(shape[2]) if shape[2] > 0 else 224
    return h, w

def _get_hw_from_keras(model):
    s = model.inputs[0].shape  # (None, H, W, 3)
    h = int(s[1]) if s[1] is not None else 224
    w = int(s[2]) if s[2] is not None else 224
    return h, w

# ===================== 러너들 =====================
class TFLiteRunner:
    def __init__(self, path, has_pre, class_list, name):
        self.cls = class_list
        self.has_pre = has_pre
        self.name = name
        self.model_path = path
        self.interp = tf.lite.Interpreter(model_path=path, num_threads=TFLITE_THREADS)
        self.interp.allocate_tensors()
        self.in_det  = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.in_idx  = self.in_det['index']
        self.out_idx = self.out_det['index']
        self.H, self.W = _get_hw_from_tflite(self.interp)
        self.in_dtype  = self.in_det['dtype']
        self.out_dtype = self.out_det['dtype']
        self.in_q  = self.in_det.get('quantization', (0.0, 0))
        self.out_q = self.out_det.get('quantization', (0.0, 0))
        print(f"[LOAD] {name}: input={self.H}x{self.W}, in_dtype={self.in_dtype}, out_dtype={self.out_dtype}")

    def _resize(self, X01, H=None, W=None):
        H = self.H if H is None else H
        W = self.W if W is None else W
        if X01.shape[1:3] == (H, W): return X01
        return tf.image.resize(X01, (H, W)).numpy()

    def _quantize_in(self, x, in_dtype=None, in_q=None):
        in_dtype = in_dtype if in_dtype is not None else self.in_dtype
        in_q = in_q if in_q is not None else self.in_q
        if in_dtype == np.float32: return x.astype(np.float32)
        scale, zp = in_q
        if scale == 0: return x.astype(in_dtype)
        q = np.round(x / scale + zp)
        if in_dtype == np.uint8: q = np.clip(q, 0, 255).astype(np.uint8)
        else: q = np.clip(q, -128, 127).astype(np.int8)
        return q

    def _dequantize_out(self, y, out_dtype=None, out_q=None):
        out_dtype = out_dtype if out_dtype is not None else self.out_dtype
        out_q = out_q if out_q is not None else self.out_q
        if out_dtype == np.float32: return y.astype(np.float32)
        scale, zp = out_q
        return ((y.astype(np.float32) - zp) * scale).astype(np.float32)

    def _invoke_once(self, interp, in_idx, out_idx, xq, out_dtype, out_q):
        interp.set_tensor(in_idx, xq)
        interp.invoke()
        y = interp.get_tensor(out_idx)
        y = self._dequantize_out(y, out_dtype, out_q)
        if not (0.0 <= y.min() and y.max() <= 1.0):
            y = 1.0/(1.0+np.exp(-y))
        return y

    def predict(self, X01):
        tag = self.name
        Xr = self._resize(X01)
        Xin = Xr.astype(np.float32) if self.has_pre else (Xr*255.0).astype(np.float32)
        N = len(Xin)
        P = np.zeros((N, len(self.cls)), dtype=np.float32)
        print(f"[RUN] {tag}: evaluating {N} images...")
        for i in range(N):
            xq = self._quantize_in(Xin[i:i+1])
            self.interp.set_tensor(self.in_idx, xq)
            self.interp.invoke()
            y = self.interp.get_tensor(self.out_idx)[0]
            y = self._dequantize_out(y)
            if not (0.0 <= y.min() and y.max() <= 1.0):
                y = 1.0/(1.0+np.exp(-y))
            P[i] = y
            if N >= 100 and (i % max(1, N//100) == 0):
                print(f"[{tag}] {i}/{N}", end='\r')
        print()
        return P

    def time_one(self, img01, warmup=TIMING_WARMUP, repeats=TIMING_REPEATS):
        h, w = self.H, self.W
        in_dtype, in_q = self.in_dtype, self.in_q
        out_dtype, out_q = self.out_dtype, self.out_q
        for _ in range(warmup):
            img01r = tf.image.resize(img01[None, ...], (h, w)).numpy()[0]
            Xin = img01r.astype(np.float32) if self.has_pre else (img01r*255.0).astype(np.float32)
            xq = self._quantize_in(Xin[None, ...], in_dtype=in_dtype, in_q=in_q)
            _ = self._invoke_once(self.interp, self.in_idx, self.out_idx, xq, out_dtype, out_q)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            if INCLUDE_RESIZE_IN_TIMING:
                img01r = tf.image.resize(img01[None, ...], (h, w)).numpy()[0]
            else:
                img01r = self._resize(img01[None, ...], h, w)[0]
            Xin = img01r.astype(np.float32) if self.has_pre else (img01r*255.0).astype(np.float32)
            xq = self._quantize_in(Xin[None, ...], in_dtype=in_dtype, in_q=in_q)
            _ = self._invoke_once(self.interp, self.in_idx, self.out_idx, xq, out_dtype, out_q)
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

class KerasRunner:
    def __init__(self, model, kind, class_list, name, has_pre=False, used_path=None):
        self.model = model
        self.kind  = kind            # 'vgg' / 'mbv2'
        self.cls   = class_list
        self.name  = name
        self.has_pre = has_pre
        self.used_path = used_path
        self.H, self.W = _get_hw_from_keras(model)
        print(f"[LOAD] {name}: input={self.H}x{self.W}, has_pre={self.has_pre}")

    def _resize(self, X01, H=None, W=None):
        H = self.H if H is None else H
        W = self.W if W is None else W
        if X01.shape[1:3] == (H, W): return X01
        return tf.image.resize(X01, (H, W)).numpy()

    def _pre(self, X01, kind=None, has_pre=None):
        kind = self.kind if kind is None else kind
        has_pre = self.has_pre if has_pre is None else has_pre
        Xr = self._resize(X01)
        if has_pre:
            return Xr.astype(np.float32)
        x = (Xr * 255.0).astype(np.float32)
        return vgg_pre(x) if kind=='vgg' else mbv2_pre(x)

    def predict(self, X01):
        tag = self.name
        N = len(X01)
        P = np.zeros((N, len(self.cls)), dtype=np.float32)
        print(f"[RUN] {tag}: evaluating {N} images (batch={KERAS_BATCH})...")
        for i in range(0, N, KERAS_BATCH):
            xb = self._pre(X01[i:i+KERAS_BATCH])
            yb = self.model(xb, training=False).numpy()
            if not (0.0 <= yb.min() and yb.max() <= 1.0):
                yb = 1.0/(1.0+np.exp(-yb))
            P[i:i+len(yb)] = yb.astype(np.float32)
            if N >= 100:
                print(f"[{tag}] {min(i+len(yb), N)}/{N}", end='\r')
        print()
        return P

    def time_one(self, img01, warmup=TIMING_WARMUP, repeats=TIMING_REPEATS):
        h, w = self.H, self.W
        for _ in range(warmup):
            img01r = tf.image.resize(img01[None, ...], (h, w)).numpy()
            xb = img01r.astype(np.float32) if self.has_pre else (
                vgg_pre((img01r*255.0).astype(np.float32)) if self.kind=='vgg'
                else mbv2_pre((img01r*255.0).astype(np.float32))
            )
            _ = self.model(xb, training=False).numpy()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            if INCLUDE_RESIZE_IN_TIMING:
                img01r = tf.image.resize(img01[None, ...], (h, w)).numpy()
            else:
                img01r = self._resize(img01[None, ...], h, w)
            xb = img01r.astype(np.float32) if self.has_pre else (
                vgg_pre((img01r*255.0).astype(np.float32)) if self.kind=='vgg'
                else mbv2_pre((img01r*255.0).astype(np.float32))
            )
            _ = self.model(xb, training=False).numpy()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

# ===================== 평가 =====================
def evaluate(name, runner, X01, Y_full, files):
    idxs = [FULL_CLASSES.index(c) for c in runner.cls]
    Y = Y_full[:, idxs]

    # 확률 예측
    P = runner.predict(X01)
    if P.shape[1] != len(runner.cls):
        raise ValueError(f"{name}: 출력 차원({P.shape[1]}) != 클래스 수({len(runner.cls)})")

    # per-class AUPRC 평균 (참고용)
    prcs=[]
    for c in range(P.shape[1]):
        try: prcs.append(average_precision_score(Y[:,c], P[:,c]))
        except: prcs.append(np.nan)
    auprc_avg = float(np.nanmean(prcs))*100
    print(f"\n== [{name}] ==")
    print(f"AUPRC (avg, per-class mean):     {auprc_avg:.2f}%")

    # 단일 threshold = 0.8
    thr = THRESHOLD
    Yb = (P >= thr).astype(np.int32)
    f1_micro  = f1_score(Y, Yb, average='micro', zero_division=0)*100
    prec      = precision_score(Y, Yb, average='micro', zero_division=0)*100
    rec       = recall_score(Y, Yb, average='micro', zero_division=0)*100

    mcm = multilabel_confusion_matrix(Y, Yb)
    specs = []
    for k in range(mcm.shape[0]):
        tn, fp = mcm[k][0,0], mcm[k][0,1]
        specs.append(tn / (tn + fp + 1e-12))
    spec_avg = float(np.mean(specs))*100

    print(f"\n-- Metrics @ threshold={thr:.2f} --")
    print(f"F1 Score (micro):                {f1_micro:.2f}%")
    print(f"Precision (micro):               {prec:.2f}%")
    print(f"Recall (micro):                  {rec:.2f}%")
    print(f"Specificity (avg):               {spec_avg:.2f}%")

    # --- per-class report (precision/recall/specificity/auprc/f1) ---
    per_cls = per_class_report(Y, P, Yb, runner.cls)
    print("\n[Per-class metrics] (precision / recall / specificity / auprc / f1, %)")
    print(per_cls.to_string(index=False))

    # 샘플 예측 표시
    print("\n========== Sample Predictions ==========")
    N = len(X01)
    for idx in random.sample(range(N), k=min(2, N)):
        print(f"[{idx:5d}] File: {os.path.basename(files[idx])}")
        print(f"     GT      : {[runner.cls[i] for i,v in enumerate(Y[idx]) if v==1]}")
        print(f"     Predict : {[runner.cls[i] for i,v in enumerate(Yb[idx]) if v==1]}  (thr={thr:.2f})\n")

# ===================== 헬퍼 =====================
def ensure_keras_model_with_path(pth):
    if os.path.isdir(pth):
        try:
            m = tf.keras.models.load_model(pth)
            return m, pth
        except Exception:
            pass
        h5s = sorted(glob.glob(os.path.join(pth, "*.h5")))
        if not h5s:
            raise FileNotFoundError(f"Keras(.h5/SavedModel) 없음: {pth}")
        used = h5s[-1]
        return load_model(used), used
    return load_model(pth), pth

def find_in_dir(dir_path, patterns):
    for pat in patterns:
        p = os.path.join(dir_path, pat)
        if os.path.exists(p): return p
        g = glob.glob(p)
        if g: return g[0]
    return ''

# ===================== 메인 =====================
def main():
    X01, files, Y_full = load_test(TEST_NPY, TEST_CSV)
    single_img_raw = load_image_raw(os.path.join(ROOT_DIR, FILENAME))

    # ---------- VGG 전용 필터: disease21=1인 샘플 제거 ----------
    idx_d21 = FULL_CLASSES.index('disease21')
    mask_vgg = (Y_full[:, idx_d21] == 0)
    X01_vgg = X01[mask_vgg]
    Y_full_vgg = Y_full[mask_vgg]
    files_vgg = [f for f, m in zip(files, mask_vgg) if m]
    print(f"[INFO] VGG eval set: {mask_vgg.sum()}/{len(mask_vgg)} samples (removed {(~mask_vgg).sum()} with disease21=1)")
    # ------------------------------------------------------------

    # 1) VGG (7클래스)
    vgg_tfl = TFLiteRunner(TFLITE_VGG, has_pre=VGG_TFLITE_HAS_PRE, class_list=LEGACY_CLASSES, name="VGG16 (TFLite)")
    evaluate("VGG16 (TFLite)", vgg_tfl, X01_vgg, Y_full_vgg, files_vgg)
    t_vgg_tfl = vgg_tfl.time_one(single_img_raw)
    size_vgg_tfl_mb = bytes_to_mb(path_size_bytes(TFLITE_VGG))

    vgg_model, vgg_used_path = ensure_keras_model_with_path(KERAS_VGG)
    vgg_ker = KerasRunner(vgg_model, kind='vgg', class_list=LEGACY_CLASSES, name="VGG16 (Keras)", has_pre=VGG_KERAS_HAS_PRE, used_path=vgg_used_path)
    evaluate("VGG16 (Keras)", vgg_ker, X01_vgg, Y_full_vgg, files_vgg)
    t_vgg_ker = vgg_ker.time_one(single_img_raw)
    size_vgg_ker_mb = bytes_to_mb(path_size_bytes(vgg_ker.used_path))

    # 2) MobileNetV2 (8클래스)
    mbv2_tfl_path = find_in_dir(MBV2_DIR, [
        "mobilenet_8_int8.tflite",
        "mobilenet_8_float16.tflite",
        "mobilenet_8_float32.tflite",
        "*.tflite"
    ])
    if not mbv2_tfl_path:
        raise FileNotFoundError(f"MobileNetV2 TFLite 없음: {MBV2_DIR}")
    mbv2_tfl = TFLiteRunner(mbv2_tfl_path, has_pre=MBV2_TFLITE_HAS_PRE, class_list=FULL_CLASSES, name="MobileNetV2 (TFLite)")
    evaluate("MobileNetV2 (TFLite)", mbv2_tfl, X01, Y_full, files)
    t_mb_tfl = mbv2_tfl.time_one(single_img_raw)
    size_mbv2_tfl_mb = bytes_to_mb(path_size_bytes(mbv2_tfl_path))

    mbv2_keras_path = find_in_dir(MBV2_DIR, ["saved_model_export","*.h5"])
    if not mbv2_keras_path:
        raise FileNotFoundError(f"MobileNetV2 Keras(.h5/SavedModel) 없음: {MBV2_DIR}")
    mbv2_model, mbv2_used_path = ensure_keras_model_with_path(mbv2_keras_path)
    mbv2_ker = KerasRunner(mbv2_model, kind='mbv2', class_list=FULL_CLASSES, name="MobileNetV2 (Keras)", has_pre=MBV2_KERAS_HAS_PRE, used_path=mbv2_used_path)
    evaluate("MobileNetV2 (Keras)", mbv2_ker, X01, Y_full, files)
    t_mb_ker = mbv2_ker.time_one(single_img_raw)
    size_mbv2_ker_mb = bytes_to_mb(path_size_bytes(mbv2_ker.used_path))

    # ===== 요약 =====
    print("\n================ Summary ================")
    print(f"Timing: resize/preprocess (and quantize for TFLite) INCLUDED, model load EXCLUDED | warmup={TIMING_WARMUP}, repeats={TIMING_REPEATS}")
    print("Per-Image Processing Time")
    print(f"  VGG16 (TFLite)       : {t_vgg_tfl*1000:.2f} ms/img")
    print(f"  VGG16 (Keras)        : {t_vgg_ker*1000:.2f} ms/img")
    print(f"  MobileNetV2 (TFLite) : {t_mb_tfl*1000:.2f} ms/img")
    print(f"  MobileNetV2 (Keras)  : {t_mb_ker*1000:.2f} ms/img")

    print("\nModel Size")
    print(f"  VGG16 (TFLite)       : {size_vgg_tfl_mb:.2f} MB  [{TFLITE_VGG}]")
    print(f"  VGG16 (Keras)        : {size_vgg_ker_mb:.2f} MB  [{vgg_ker.used_path}]")
    print(f"  MobileNetV2 (TFLite) : {size_mbv2_tfl_mb:.2f} MB  [{mbv2_tfl_path}]")
    print(f"  MobileNetV2 (Keras)  : {size_mbv2_ker_mb:.2f} MB  [{mbv2_ker.used_path}]")
    print("=========================================\n")

if __name__ == "__main__":
    main()
