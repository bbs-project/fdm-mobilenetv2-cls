
# ###########################기존 모델 학습 당시 data로 test########################
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
from sklearn.metrics import f1_score

# ======== 경로 설정 ========
tflite_model_path = '/home/user/fdm/lite-models/tflite/fdm_vgg16.tflite'
image_npy_path = '/home/user/fdm/rgb-classify-vgg16/data/npy/test.npy'
label_csv_path = '/home/user/fdm/rgb-classify-vgg16/data/label/test.csv'

# ======== 데이터 로드 ========
x_test = np.load(image_npy_path)
label_df = pd.read_csv(label_csv_path)

# ======== TFLite 모델 로드 ========
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

output_shape = output_details[0]['shape']
num_classes = output_shape[-1]
print(f"🔎 Output shape: {output_shape} → 예측 클래스 수: {num_classes}")
print("TFLite 모델 입력 shape:", input_details[0]['shape'])
print("x_test shape:", x_test.shape)

# ======== 라벨 전처리 (파일명 제외하고 클래스 열만 선택) ========
y_test = label_df.iloc[:, 1:1 + num_classes].values  # 파일명 제외, 모델 클래스 수에 맞게 자름

# ======== 추론 + 시간 측정 ========
preds = []
inference_times = []

# 진행 상황 출력용 변수
total_images = len(x_test)
processed_images = 0

start_time = time.time()

for i in range(total_images):
    input_data = x_test[i:i+1].astype(np.float32)
    inference_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    inference_end = time.time()

    preds.append(output[0])
    inference_times.append(inference_end - inference_start)

    # 진행률 출력
    processed_images += 1
    if processed_images % 100 == 0:
        elapsed_time = time.time() - start_time
        estimated_remaining_time = (elapsed_time / processed_images) * (total_images - processed_images)
        print(f" {processed_images}/{total_images} 이미지 처리 완료")
        print(f" 예상 남은 시간: {estimated_remaining_time / 60:.2f}분")

# 예측을 이진 클래스(0과 1)로 변환
preds = np.array(preds)
preds_bin = (preds > 0.5).astype(int)

# ======== 평가 지표 출력 ========
f1 = f1_score(y_test, preds_bin, average='micro')
model_size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
avg_time = np.mean(inference_times)
total_time = np.sum(inference_times)

print(f"\n F1-score (micro): {f1:.4f}")
print(f" 평균 추론 시간 per image: {avg_time:.6f} sec")
print(f" 전체 추론 시간: {total_time:.2f} sec")
print(f" 모델 용량: {model_size_mb:.2f} MB")

