import pandas as pd
import os

# 경로 설정
base_dir = "/home/user/chae_project/vgg_data/112_after0708/labels"
backup_dir = os.path.join(base_dir, "class13")  # 기존 파일 위치
csv_files = ["train.csv", "valid.csv", "test.csv"]

# 통합 대상 열
merge_cols = ["disease14", "disease16", "disease17", "disease18"]
target_col = "disease13"

for file in csv_files:
    backup_path = os.path.join(backup_dir, file)  # 기존 파일 경로
    save_path = os.path.join(base_dir, file)      # 새로 저장할 경로

    # CSV 불러오기
    df = pd.read_csv(backup_path)
    
    # disease13에 병합할 컬럼들의 값을 더함
    df[target_col] = df[[target_col] + merge_cols].sum(axis=1)
    
    # 값이 1 이상이면 1, 아니면 0으로 처리
    df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
    
    # 병합 대상 컬럼 삭제
    df = df.drop(columns=merge_cols)
    
    # 새 파일로 저장
    df.to_csv(save_path, index=False)
    print(f"{file} 저장 완료 → {save_path}")
