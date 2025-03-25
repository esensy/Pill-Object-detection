import json
import os
import argparse
import shutil
import yaml
from src.data_utils.data_loader import get_category_mapping
from sklearn.model_selection import train_test_split

# COCO JSON -> YOLO TXT 변환 함수
def convert_json_to_txt(json_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ANN_DIR = "data/train_annots_modify"
    name_to_idx, _ = get_category_mapping(ANN_DIR)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 각 이미지에 대해
    for img in data["images"]:
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        # .png -> .txt
        label_path = os.path.join(output_dir, f"{img['file_name'].replace('.png', '.txt')}")

        # 해당 이미지의 annotation을 찾아 YOLO format으로 저장
        with open(label_path, "w", encoding="utf-8") as f:
            for ann in data["annotations"]:
                if ann["image_id"] == img_id:
                    x, y, w, h = ann["bbox"]
                    x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
                    w, h = w / img_w, h / img_h

                    # category_id 매칭 및 YOLO 형식 라벨 작성
                    for category in data['categories']:
                        if ann["category_id"] == category["id"]:
                            # 0:배경을 제거하기 위해서 -1
                            category_id = name_to_idx[category['name']] - 1
                            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# 폴더 내 모든 JSON을 변환
def process_all_json(json_folder, output_dir):
    if not os.path.exists(json_folder):
        print(f"JSON 폴더가 존재하지 않습니다: {json_folder}")
        return
    
    # output_dir 폴더가 없으면 생성 (있으면 무시)
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if len(json_files) == 0:
        print("변환할 JSON 파일이 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 각 JSON 파일 변환 호출
    for i, json_file in enumerate(json_files, start=1):
        json_path = os.path.join(json_folder, json_file)
        convert_json_to_txt(json_path, output_dir)


    print("모든 JSON 파일 변환 완료")

# 라벨과 이미지 학습-검증 분할 및 복사 함수
def split_labels_and_images(label_dir, image_dir, output_train, output_val, test_size=0.2, random_state=42):
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)

    # 라벨 파일 목록 가져오기
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    print(f"라벨 파일 개수: {len(label_files)}")
    if len(label_files) == 0:
        print(f"경로 {label_dir}에 .txt 라벨 파일이 없습니다. 경로를 다시 확인하세요.")
        return

    # stratify용 class id 추출
    class_ids = []
    for filename in label_files:
        file_path = os.path.join(label_dir, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 마지막 줄 사용
            if len(lines) == 4:
                # 4번째 줄의 첫 번째 값
                class_id = int(lines[3].split()[0])
            elif len(lines) == 3:
                # 3번째 줄의 첫 번째 값
                class_id = int(lines[2].split()[0])
            else:
                # 비어 있는 파일은 0 처리
                class_id = 0
        class_ids.append(class_id)

    # train/val 분할 (stratify 시도 후 실패 시 랜덤 분할)
    try:
        train_labels, val_labels = train_test_split(
            label_files, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=class_ids
        )
        print(f"Stratify 성공, 마지막 class id를 기준으로 분할합니다.")
    except Exception as e:
        print(f"Stratify 실패, 랜덤 분할로 대체합니다. 오류: {e}")
        train_labels, val_labels = train_test_split(
            label_files, 
            test_size=test_size, 
            random_state=random_state
        )


    # train 라벨 및 이미지 복사
    for label_file in train_labels:
        shutil.move(os.path.join(label_dir, label_file), os.path.join(output_train, label_file))
        img_name = label_file.replace('.txt', '.png')
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_train, img_name)
        shutil.copy(src_img_path, dst_img_path)

    # val 라벨 및 이미지 복사
    for label_file in val_labels:
        shutil.move(os.path.join(label_dir, label_file), os.path.join(output_val, label_file))
        img_name = label_file.replace('.txt', '.png')
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_val, img_name)
        shutil.copy(src_img_path, dst_img_path)

    print("라벨 및 이미지 분할 완료.")



# data.yaml 파일 생성 함수
def make_yaml(train_dir, val_dir, output_dir):
    
    # 경로 준비
    os.makedirs(output_dir, exist_ok=True)

    # YAML 파일 경로 설정
    yaml_dir = os.path.join(output_dir, "data.yaml")
    
    # 클래스 이름 목록 가져오기
    _, idx_to_name = get_category_mapping("data/train_annots_modify") 
    keys = list(idx_to_name.keys())  
    del idx_to_name[keys[0]], idx_to_name[keys[-1]]   # 마지막 키: ㄴno_class, 첫번째 키: ㄴbackground 삭제
    class_names = [name for name in idx_to_name.values()]

    # names 항목을 한 줄 문자열 포맷으로 변환
    formatted_names = "[" + ", ".join([f'\"{name}\"' for name in class_names]) + "]"

    # YAML 내용 문자열 직접 작성 (따옴표 및 포맷 유지)
    yaml_content = f"""
train: {train_dir}
val: {val_dir}
nc: {len(class_names)}
names: {formatted_names}
"""

    try:
        with open(yaml_dir, "w", encoding='utf-8') as f:
            f.write(yaml_content.strip())
        print(f"YAML 파일이 '{yaml_dir}'에 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"YAML 파일 생성 중 오류가 발생했습니다: {e}")

    print(f"{yaml_dir} 파일이 생성되었습니다.")

###########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format")
    parser.add_argument("--json_folder", type=str, default="data/train_annots_modify", help="Folder containing COCO JSON files")
    parser.add_argument("--output_dir", type=str, default="data/train_labels", help="Output directory for YOLO label files")
    args = parser.parse_args()

    #process_all_json(args.json_folder, args.output_dir)

    
    label_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels"
    image_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_images"
    output_train = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\train"
    output_val = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\val"
    #split_labels_and_images(label_dir, image_dir, output_train, output_val, test_size=0.2, random_state=42)

    train_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\train"
    val_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\val"
    output_dir = r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels"
    make_yaml(train_dir, val_dir, output_dir)