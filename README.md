# MobileNetV2-based Flatfish Disease Classfication Model
* Started from VGG16 model
* Changed to MobileNetV2 model
* Applied some optimization, augmentation skills.
* Generated and Tested a TFLite model for mobile devices.

## 0. conda 가상 환경 설정
```
$ conda activate vgg16
```

## 1. 모델 train 및 test
/home/user/chae_project/mobile
안에 각각 train, test 파일로 있음
 - 모델은 5~8까지 총 4번에 걸쳐서 학습 진행. (5>6->7>8 ;이어학습)
 - 최종 모델 : mobilenet_8

## 2. 저장된 모델
/home/user/chae_project/model_save
안에 각각 모델마다의 파일 존재
 - `best_model.*` : 성능이 가장 좋았던 시점의 모델 가중치
 - `saved_weights.*` : 마지막 epoch 학습이 끝난 시점의 모델 가중치
 - `checkpoint` : TensorFlow 체크포인트 관리 파일
 - `best_threshold.npy` : 클래스별 최적 threshold 값
 - `temp_scale.npy` : temperature scaling 보정 값

 mobilenet_8의 경우 
 - .h5파일 존재
 - TFLite 변환 파일 존재

## 3. 사용 데이터
/home/user/chae_project/vgg_data
각각 train / test / valid 에 대해 npy와 csv 파일 존재
 - npy : 224 x 224 사이즈
 - csv : class는 8개 존재 (disease1,disease2,disease6,disease8,disease11,disease13,disease19,disease21)
         비브리오 병은 하나로 통일

## 4. 기존 VGG TFLite
/home/user/chae_project/VGG16_TFLite
 - tplite_250814.py : 현재 data로 test
 - tplite.py : 기존 모델 학습 당시 data로 test

## 5. 기타
### (1)논문 평가지표
/home/user/chae_project/all_test.py
 - F1 score, Precision, Recall, Specificity, AUPRC
 - 각 class별 
 - 한 장당 처리 시간 및 모델 크기 측정

### (2) threshold별 F1 score
/home/user/chae_project/threshold_F1.py
 - 0.1 부터 0.9까지 F1 비교
