import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil
import time

def save_npy_files(image_dir, label_dir, output_npy_path, file_names):
    images = []
    
    for idx, file_name in enumerate(file_names):
        image_path = os.path.join(image_dir, file_name + '.JPG')  # 이미지 파일 경로
        label_path = os.path.join(label_dir, file_name + '.json')  # 라벨 파일 경로
        
        # 이미지 파일 로드 및 전처리
        image = load_img(image_path, target_size=(224, 224))  # VGG16에 맞게 크기 조정
        image_array = img_to_array(image) / 255.0  # 정규화
        
        # 이미지 저장
        images.append(image_array)
        
        # 100번마다 시간과 진행상황 출력
        if (idx + 1) % 100 == 0 or idx + 1 == len(file_names):
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / (idx + 1)
            remaining_images = len(file_names) - (idx + 1)
            remaining_time = avg_time_per_image * remaining_images

            # 시간을 h:m:s 형식으로 변환
            elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

            print(f"Processing image {idx + 1}/{len(file_names)}")
            print(f"지나간 시간: {elapsed_time_hms} (hh:mm:ss).")
            print(f"남은시간: {remaining_time_hms} (hh:mm:ss).")

    # npy 파일로 저장
    images_array = np.array(images)
    np.save(output_npy_path, images_array)

def split_and_process_data(image_dir, label_dir, output_image_dir, output_label_dir, test_size=0.2):
    # 이미지 및 라벨 파일 목록
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    # 이미지 파일명 추출 (확장자 제외)
    image_names = set(os.path.splitext(f)[0] for f in image_files)
    
    # Train과 test 데이터로 분리 (80% train, 20% test)
    train_files, test_files = train_test_split(list(image_names), test_size=test_size, random_state=42)
    
    # Train, test 파일 경로 출력
    print(f"Total images: {len(image_names)}")
    print(f"Train images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")

    # 시간 측정 시작
    global start_time
    start_time = time.time()

    # Train 이미지 npy로 저장
    save_npy_files(image_dir, label_dir, os.path.join(output_image_dir, 'train.npy'), train_files)
    
    # Test 이미지 npy로 저장
    save_npy_files(image_dir, label_dir, os.path.join(output_image_dir, 'test.npy'), test_files)
    
    # 라벨 파일을 train과 test로 분리하여 복사
    os.makedirs(os.path.join(output_label_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_label_dir, 'test'), exist_ok=True)

    # train과 test 디렉토리에 라벨 복사
    for file_name in train_files:
        label_path = os.path.join(label_dir, file_name + '.json')
        dest_label_path = os.path.join(output_label_dir, 'train', file_name + '.json')
        shutil.copy(label_path, dest_label_path)
    
    for file_name in test_files:
        label_path = os.path.join(label_dir, file_name + '.json')
        dest_label_path = os.path.join(output_label_dir, 'test', file_name + '.json')
        shutil.copy(label_path, dest_label_path)

    print(f"Train and Test NPY files have been saved to {output_image_dir}")
    print(f"Train and Test label files have been copied to {output_label_dir}")

# 경로 설정
image_dir = '/data/301.FlounderDiseaseData-2024.03/01-1.정식개방데이터/Training/01.원천데이터/rgb'
label_dir = '/data/301.FlounderDiseaseData-2024.03/01-1.정식개방데이터/Training/02.라벨링데이터/rgb'
output_image_dir = '/home/user/chae_project/vgg_data/'
output_label_dir = '/home/user/chae_project/vgg_data/labels'

# 데이터 분할 및 NPY 저장
split_and_process_data(image_dir, label_dir, output_image_dir, output_label_dir)
