# f1_sweep_only.py
import os, glob, random
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mbv2_pre

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===================== 경로/설정 =====================
DATASET_PATH = "/home/user/chae_project/vgg_data/0805_224/"
TEST_NPY = os.path.join(DATASET_PATH, "test.npy")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")

# 모델 경로
TFLITE_VGG = "/home/user/fdm/lite-models/tflite/fdm_vgg16.tflite"         # (입력 112 가정)
KERAS_VGG  = "/home/user/fdm/rgb-classify-vgg16/saved_model/my_model"     # 폴더 안 .h5 또는 SavedModel
MBV2_DIR   = "/home/user/chae_project/model_save/mobilenet_8"             # 8번 모델 폴더

# 모바일넷 per-class threshold 파일(네가 준 경로)
VEC_THR_PATH = "/home/user/chae_project/model_save/mobilenet_8/best_threshold.npy"

# 클래스
FULL_CLASSES   = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19','disease21']  # 8 (MBV2)
LEGACY_CLASSES = ['disease1','disease2','disease6','disease8','disease11','disease13','disease19']              # 7 (VGG: disease21 제외)

# 전처리 플래그
VGG_TFLITE_HAS_PRE  = True
MBV2_TFLITE_HAS_PRE = True
VGG_KERAS_HAS_PRE   = True
MBV2_KERAS_HAS_PRE  = True

# 실행 파라미터
KERAS_BATCH = 64
THR_SWEEP = np.round(np.arange(0.1, 1.0, 0.1), 2)

# ===================== 유틸 =====================
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

def _get_hw_from_tflite(interp):
    det = interp.get_input_details()[0]
    shape = det.get('shape_signature', det['shape'])
    h = int(shape[1]) if shape[1] > 0 else 224
    w = int(shape[2]) if shape[2] > 0 else 224
    return h, w

def _get_hw_from_keras(model):
    s = model.inputs[0].shape
    h = int(s[1]) if s[1] is not None else 224
    w = int(s[2]) if s[2] is not None else 224
    return h, w

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

def load_and_check_thr_vec(path, class_names):
    thr = np.load(path)
    print(f"[LOAD] best_threshold.npy: shape={thr.shape}, dtype={thr.dtype}")
    if thr.ndim != 1:
        raise ValueError(f"threshold 벡터가 1차원이 아님: {thr.shape}")
    if len(thr) != len(class_names):
        raise ValueError(f"threshold 길이 불일치: vec={len(thr)} vs classes={len(class_names)}")
    if not (np.all(thr >= 0.0) and np.all(thr <= 1.0)):
        raise ValueError(f"threshold 값 범위 이상(0~1 아님): min={thr.min():.3f}, max={thr.max():.3f}")
    print(f"[OK] matched classes={len(class_names)} | min={thr.min():.2f}, max={thr.max():.2f}")
    return thr.astype(np.float32)

# ===================== 러너들 =====================
class TFLiteRunner:
    def __init__(self, path, has_pre, class_list, name):
        self.cls = class_list
        self.has_pre = has_pre
        self.name = name
        self.model_path = path
        self.interp = tf.lite.Interpreter(model_path=path)
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

    def _quantize_in(self, x):
        if self.in_dtype == np.float32: return x.astype(np.float32)
        scale, zp = self.in_q
        if scale == 0: return x.astype(self.in_dtype)
        q = np.round(x / scale + zp)
        if self.in_dtype == np.uint8: q = np.clip(q, 0, 255).astype(np.uint8)
        else: q = np.clip(q, -128, 127).astype(np.int8)
        return q

    def _dequantize_out(self, y):
        if self.out_dtype == np.float32: return y.astype(np.float32)
        scale, zp = self.out_q
        return ((y.astype(np.float32) - zp) * scale).astype(np.float32)

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
            if N >= 100 and i % (N//100) == 0:
                print(f"[{tag}] {i}/{N}", end='\r')
        return P

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

    def _pre(self, X01):
        Xr = self._resize(X01)
        if self.has_pre:
            return Xr.astype(np.float32)
        x = (Xr * 255.0).astype(np.float32)
        return vgg_pre(x) if self.kind=='vgg' else mbv2_pre(x)

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
                print(f"[{tag}] {min(i+len(yb),N)}/{N}", end='\r')
        return P

# ===================== F1 평가 =====================
def f1_micro_at_thresholds(Y, P, thr_list):
    rows=[]
    for t in thr_list:
        Yb=(P>=t).astype(np.int32)
        rows.append((t, round(f1_score(Y, Yb, average='micro', zero_division=0)*100,2)))
    best=max(rows,key=lambda x:x[1])
    return rows,best

def f1_micro_with_vector_thresholds(Y, P, thr_vec):
    Yb=(P>=thr_vec.reshape(1,-1)).astype(np.int32)
    return f1_score(Y, Yb, average='micro', zero_division=0)*100

def evaluate_f1_sweep(name, runner, X01, Y_full, vec_thr_path=None):
    idxs=[FULL_CLASSES.index(c) for c in runner.cls]
    Y=Y_full[:,idxs]
    P=runner.predict(X01)

    print(f"\n== [{name}] micro-F1 sweep (0.1~0.9) ==")
    print(" threshold  f1_micro(%)")
    rows,best=f1_micro_at_thresholds(Y,P,THR_SWEEP)
    for t,f1m in rows: print(f"      {t:.1f}        {f1m:.2f}")
    print(f"[BEST] thr={best[0]:.2f} → F1(micro)={best[1]:.2f}%")

    # 모바일넷 둘 다에 per-class threshold 적용 (있으면)
    if vec_thr_path:
        try:
            thr_vec = load_and_check_thr_vec(vec_thr_path, runner.cls)
            f1v = f1_micro_with_vector_thresholds(Y,P,thr_vec)
            print(f"F1(micro) @ vector thresholds = {f1v:.2f}%")
        except Exception as e:
            print(f"[Per-class thresholds] 사용 실패: {e}")

# ===================== 메인 =====================
def main():
    X01, files, Y_full = load_test(TEST_NPY, TEST_CSV)

    # ---------- VGG 전용: disease21=1 샘플 제거 ----------
    idx_d21 = FULL_CLASSES.index('disease21')
    mask_vgg = (Y_full[:, idx_d21] == 0)
    X01_vgg = X01[mask_vgg]
    Y_full_vgg = Y_full[mask_vgg]
    print(f"[INFO] VGG eval set: {mask_vgg.sum()}/{len(mask_vgg)} samples (removed {(~mask_vgg).sum()} with disease21=1)")
    # -----------------------------------------------------

    # 1) VGG (TFLite)
    if os.path.exists(TFLITE_VGG):
        vgg_tfl = TFLiteRunner(TFLITE_VGG, has_pre=VGG_TFLITE_HAS_PRE, class_list=LEGACY_CLASSES, name="VGG16 (TFLite)")
        evaluate_f1_sweep("VGG16 (TFLite)", vgg_tfl, X01_vgg, Y_full_vgg, vec_thr_path=None)
    else:
        print("[SKIP] VGG16 (TFLite) not found")

    # 2) VGG (Keras)
    try:
        vgg_model, vgg_used_path = ensure_keras_model_with_path(KERAS_VGG)
        vgg_ker = KerasRunner(vgg_model, kind='vgg', class_list=LEGACY_CLASSES, name="VGG16 (Keras)", has_pre=VGG_KERAS_HAS_PRE, used_path=vgg_used_path)
        evaluate_f1_sweep("VGG16 (Keras)", vgg_ker, X01_vgg, Y_full_vgg, vec_thr_path=None)
    except Exception as e:
        print(f"[SKIP] VGG16 (Keras) load fail: {e}")

    # 3) MobileNetV2 (TFLite) — per-class threshold 적용
    mbv2_tfl_path = find_in_dir(MBV2_DIR, [
        "mobilenet_8_int8.tflite",
        "mobilenet_8_float16.tflite",
        "mobilenet_8_float32.tflite",
        "*.tflite"
    ])
    if mbv2_tfl_path:
        mb_tfl = TFLiteRunner(mbv2_tfl_path, has_pre=MBV2_TFLITE_HAS_PRE, class_list=FULL_CLASSES, name="MobileNetV2 (TFLite)")
        evaluate_f1_sweep("MobileNetV2 (TFLite)", mb_tfl, X01, Y_full, vec_thr_path=VEC_THR_PATH)
    else:
        print("[SKIP] MobileNetV2 (TFLite) not found")

    # 4) MobileNetV2 (Keras) — per-class threshold 적용
    mbv2_keras_path = find_in_dir(MBV2_DIR, ["saved_model_export","*.h5"])
    if mbv2_keras_path:
        m, used = ensure_keras_model_with_path(mbv2_keras_path)
        mb_ker = KerasRunner(m, kind='mbv2', class_list=FULL_CLASSES, name="MobileNetV2 (Keras)", has_pre=MBV2_KERAS_HAS_PRE, used_path=used)
        evaluate_f1_sweep("MobileNetV2 (Keras)", mb_ker, X01, Y_full, vec_thr_path=VEC_THR_PATH)
    else:
        print("[SKIP] MobileNetV2 (Keras) not found")

if __name__ == "__main__":
    main()
