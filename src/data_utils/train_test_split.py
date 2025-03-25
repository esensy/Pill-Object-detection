################################################################################################
# 데이터 다운 -> data loader 실행 -> json modify 실행 -> coco to yolo -> train split yolo 순서
# 실행 코드 python train_split_yolo.py
# train_labels → train/val 분할 및 train.txt, val.txt 생성
# train, val.txt는 train과 val에 맞는 img 파일 path 정보가 들어있는 파일
# ###############################################################################################

import os
import shutil
import yaml
from src.data_utils.data_loader import get_category_mapping
from sklearn.model_selection import train_test_split

def split_labels_and_images(label_dir, image_dir, output_train, output_val, test_size=0.2, random_state=42):
    """
    라벨(.txt) 파일과 이미지(.png) 파일을 학습/검증으로 분할하는 함수.

    Args:
        label_dir (str): 원본 라벨 폴더 경로
        image_dir (str): 이미지 파일 경로
        output_train (str): 학습 라벨 저장 경로
        output_val (str): 검증 라벨 저장 경로
        train_txt_path (str): 학습 이미지 리스트 파일 경로
        val_txt_path (str): 검증 이미지 리스트 파일 경로
        test_size (float): 검증 세트 비율
        random_state (int): 난수 시드
    """

    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    print(f"라벨 파일 개수: {len(label_files)}")
    if len(label_files) == 0:
        print(f"경로 {label_dir}에 .txt 라벨 파일이 없습니다. 경로를 다시 확인하세요.")
        return

    # stratify 정보 수집
    class_ids = []
    for filename in label_files:
        file_path = os.path.join(label_dir, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
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


    # train 폴더로 이동
    for label_file in train_labels:
        shutil.move(os.path.join(label_dir, label_file), os.path.join(output_train, label_file))
        img_name = label_file.replace('.txt', '.png')
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_train, img_name)
        shutil.copy(src_img_path, dst_img_path)

    # val 폴더로 이동
    for label_file in val_labels:
        shutil.move(os.path.join(label_dir, label_file), os.path.join(output_val, label_file))
        img_name = label_file.replace('.txt', '.png')
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_val, img_name)
        shutil.copy(src_img_path, dst_img_path)

    print("라벨 및 이미지 분할 완료.")




def make_yaml(train_dir, val_dir, output_dir):
    
    # 경로 준비
    os.makedirs(output_dir, exist_ok=True)

    # YAML 파일 경로 설정
    yaml_dir = os.path.join(output_dir, "data.yaml")
    
    # 클래스 설정
    _, idx_to_name = get_category_mapping("data/train_annots_modify") 
    keys = list(idx_to_name.keys())  
    del idx_to_name[keys[0]], idx_to_name[keys[-1]]   # 마지막 키: ㄴno_class, 첫번째 키: ㄴbackground 삭제
    class_names = [name for name in idx_to_name.values()]

    #  make_yaml 함수에서 사용할 txt 파일을 생성
    image_folder ='data/train_images'
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png'))]

    # train:과 val:이 같은 폴더를 가리키므로, YOLO가 _val.txt, _train.txt을 참조해 자동으로 이미지를 찾을 수 있습니다. 
    data = {
        "train": train_dir,  
        "val": val_dir, 
        "nc": len(class_names),
        "names": class_names
    }

    try:
        with open(yaml_dir, "w", encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"YAML 파일이 '{yaml_dir}'에 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"YAML 파일 생성 중 오류가 발생했습니다: {e}")

    print(f"{yaml_dir} 파일이 생성되었습니다.")

###########################################################################################


if __name__ == "__main__":
    label_dir = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels"
    image_dir = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_images"
    output_train = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels\train"
    output_val = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels\val"
    split_labels_and_images(label_dir, image_dir, output_train, output_val, test_size=0.2, random_state=42)

    train_dir = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels\train"
    val_dir = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels\val"
    output_dir = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels"
    make_yaml(train_dir, val_dir, output_dir)