import os
import shutil

# 원본 이미지 폴더 (이미지가 저장된 곳)
orig_img_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_images"

# train_labels 폴더 내의 train과 val 경로 (여기로 이미지 파일을 복사할 예정)
train_labels_train_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\train"
train_labels_val_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\val"

def copy_matching_images(labels_dir, img_ext=".png"):
    """
    labels_dir 내에 있는 .txt 파일 이름에 맞는 이미지 파일을 원본 폴더에서 찾아서
    labels_dir로 복사합니다.
    """
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    count = 0
    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]  # 예: image1.txt -> image1
        image_filename = base_name + img_ext          # 예: image1.png
        src_img_path = os.path.join(orig_img_dir, image_filename)
        dest_img_path = os.path.join(labels_dir, image_filename)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dest_img_path)
            count += 1
        else:
            print(f"Warning: {src_img_path} not found.")
    print(f"Copied {count} images from {orig_img_dir} to {labels_dir}")

# train과 val 각각에 대해 이미지 복사
copy_matching_images(train_labels_train_dir)
copy_matching_images(train_labels_val_dir)
