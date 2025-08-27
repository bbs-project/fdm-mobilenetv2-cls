import os
import numpy as np
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 이미지 경로
image_dir = '/data/301.FlounderDiseaseData-2024.03/01-1.정식개방데이터/Validation/01.원천데이터/rgb'
save_dir = '/home/user/chae_project/vgg_data'

# 'vgg_data' 폴더가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 이미지 데이터를 저장할 리스트
images = []

# 전체 이미지 파일 수 확인
image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPG') or f.endswith('.PNG')]
total_images = len(image_files)

# 시간 측정 시작
start_time = time.time()

# 이미지 파일을 읽어서 처리
for idx, image_name in enumerate(image_files):
    image_path = os.path.join(image_dir, image_name)
    
    # 이미지 로드 및 크기 조정
    image = load_img(image_path, target_size=(224, 224))  # VGG16에 맞게 크기 조정
    image_array = img_to_array(image) / 255.0  # 정규화

    # 이미지를 배열에 추가
    images.append(image_array)

    # 진행 상태 출력 (100번째마다 걸린 시간과 남은 시간 출력)
    if (idx + 1) % 100 == 0 or idx + 1 == total_images:
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / (idx + 1)
        remaining_images = total_images - (idx + 1)
        remaining_time = avg_time_per_image * remaining_images

        # 시간을 h:m:s 형식으로 변환
        elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

        print(f"Processing image {idx + 1}/{total_images}")
        print(f"Time taken for last 100 images: {elapsed_time_hms} (hh:mm:ss).")
        print(f"Estimated remaining time: {remaining_time_hms} (hh:mm:ss).")

# 이미지를 numpy 배열로 변환
images_array = np.array(images)

# npy 파일로 저장 (train.npy로 저장)
npy_file_path = os.path.join(save_dir, 'valid.npy')
np.save(npy_file_path, images_array)

# 전체 시간 출력
end_time = time.time()
total_time = end_time - start_time
total_time_hms = time.strftime("%H:%M:%S", time.gmtime(total_time))

print(f"Data saved at {npy_file_path}")
print(f"Total time taken: {total_time_hms} (hh:mm:ss)")
