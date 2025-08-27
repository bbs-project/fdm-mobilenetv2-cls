# ###########################현재 data로 test########################

import numpy as np
import pandas as pd
import tensorflow as tf
import time, os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# ===== 경로 =====
tflite_model_path = '/home/user/fdm/lite-models/tflite/fdm_vgg16.tflite'
image_npy_path    = '/home/user/chae_project/vgg_data/test.npy'
label_csv_path    = '/home/user/chae_project/vgg_data/test.csv'

# ===== 데이터 =====
x_test = np.load(image_npy_path)
label_df = pd.read_csv(label_csv_path)

# ===== TFLite 로드 =====
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 모델 출력/입력 사양
output_shape = output_details[0]['shape']
num_classes  = int(output_shape[-1])
in_h, in_w   = int(input_details[0]['shape'][1]), int(input_details[0]['shape'][2])

print(f"Output shape: {output_shape} → 클래스 수: {num_classes}")
print("TFLite 입력 shape:", input_details[0]['shape'])
print("x_test shape (before):", x_test.shape)

# --- 입력 크기/스케일 정합화 (예: 224→112)
if x_test.shape[1] != in_h or x_test.shape[2] != in_w:
    x_test = tf.image.resize(x_test, (in_h, in_w)).numpy()

x_test = x_test.astype(np.float32)
if x_test.max() > 1.0:  # 원 데이터가 0~255라면 0~1로 정규화
    x_test /= 255.0

print("x_test shape (after):", x_test.shape)

# ===== 라벨 정렬: disease21 자동 제외 + 모델 출력 수에 맞춤 =====
label_cols = list(label_df.columns[1:])  # 첫 열이 파일명이라고 가정
if 'disease21' in label_cols:
    label_cols.remove('disease21')
label_cols = label_cols[:num_classes]
y_test = label_df[label_cols].values.astype(int)

assert y_test.shape[1] == num_classes, f"라벨 열({y_test.shape[1]}) != 모델 출력({num_classes})"

# ===== 추론 =====
preds = []
times = []
n = len(x_test)
t0 = time.time()

for i in range(n):
    inp = x_test[i:i+1]  # float32, (1, H, W, 3)
    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    t2 = time.time()

    preds.append(out)
    times.append(t2 - t1)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n - (i + 1))
        print(f" {i+1}/{n} 처리 |  남은 시간 ~ {eta/60:.2f}분")

preds = np.array(preds, dtype=np.float32)
preds_bin = (preds > 0.5).astype(int)

# ===== 지표 =====
f1_micro        = f1_score(y_test, preds_bin, average='micro', zero_division=0)
precision_micro = precision_score(y_test, preds_bin, average='micro', zero_division=0)
recall_micro    = recall_score(y_test, preds_bin, average='micro', zero_division=0)
try:
    auprc_micro = roc_auc_score(y_test, preds, average='micro')  # PR AUC를 원하면 average_precision_score 사용 가능
except Exception:
    auprc_micro = None

# --- Specificity (micro)
tp = np.sum((y_test == 1) & (preds_bin == 1))
fp = np.sum((y_test == 0) & (preds_bin == 1))
fn = np.sum((y_test == 1) & (preds_bin == 0))
tn = np.sum((y_test == 0) & (preds_bin == 0))
specificity_micro = tn / (tn + fp + 1e-7)

# --- Specificity (per-class)
specificity_per_class = []
for c in range(y_test.shape[1]):
    y_c = y_test[:, c]
    p_c = preds_bin[:, c]
    tn_c = np.sum((y_c == 0) & (p_c == 0))
    fp_c = np.sum((y_c == 0) & (p_c == 1))
    specificity_per_class.append(tn_c / (tn_c + fp_c + 1e-7))

# ===== 리소스/시간 =====
size_mb  = os.path.getsize(tflite_model_path) / (1024 * 1024)
avg_time = float(np.mean(times)) if len(times) else 0.0
sum_time = float(np.sum(times))

# ===== 출력 =====
print("\n 평가 결과")
print(f"F1-score (micro): {f1_micro:.4f}")
print(f"Precision (micro): {precision_micro:.4f}")
print(f"Recall (micro): {recall_micro:.4f}")
print(f"Specificity (micro): {specificity_micro:.4f}")
print("Specificity (per class):", [f"{s:.4f}" for s in specificity_per_class])
if auprc_micro is not None:
    print(f"AUPRC (micro): {auprc_micro:.4f}")
print(f"평균 추론 시간/이미지: {avg_time:.6f} sec")
print(f"전체 추론 시간: {sum_time:.2f} sec")
print(f"모델 용량: {size_mb:.2f} MB")
print(f"사용 라벨 열: {label_cols}")
