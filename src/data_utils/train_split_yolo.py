################################################################################################
# 데이터 다운 -> data loader 실행 -> json modify 실행 -> coco to yolo -> train split yolo 순서
# 실행 코드
# python coco_to_yolo.py --json_folder data/train_annots_modify --output_dir data/train_labels_YOLO
# category_id x_center y_center width height + 좌표 정규화
# 이렇게 바꿔놓아야 YOLO에서 돌아간다고 합니다
# ###############################################################################################

import os
import shutil
from sklearn.model_selection import train_test_split

# YOLO 데이터 경로 설정
image_dir = "data/train_images"
label_dir = "data/train_labels_YOLO"

train_image_dir = "data/train_images/train"
val_image_dir = "data/train_images/val"

train_label_dir = "data/train_labels/train"
val_label_dir = "data/train_labels/val"

# 폴더 생성
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
label_files = [f.replace('.png', '.txt') for f in image_files]  # YOLO 라벨 파일명

# Train/Val 데이터 분할 (80:20)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Train 데이터 이동
for img in train_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(train_image_dir, img))
    label = img.replace(".png", ".txt")
    if os.path.exists(os.path.join(label_dir, label)):
        shutil.move(os.path.join(label_dir, label), os.path.join(train_label_dir, label))

# Val 데이터 이동
for img in val_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(val_image_dir, img))
    label = img.replace(".png", ".txt")
    if os.path.exists(os.path.join(label_dir, label)):
        shutil.move(os.path.join(label_dir, label), os.path.join(val_label_dir, label))

print("✅ Train/Val 데이터셋 분할 및 이동 완료!")

# 분할을 하는데 이미지는 놔두고, txt만 train.txt, val.txt파일 만들는 모듈? 함수?
# 8:2로 이미지 패스들이 있어야하고, 어떤 정보들이 있어야하는지?