################################################################################################
# 데이터 다운 -> data loader 실행 -> json modify 실행 -> coco to yolo -> train split yolo 순서
# 실행 코드 python train_split_yolo.py
# train_labels → train/val 분할 및 train.txt, val.txt 생성
# train, val.txt는 train과 val에 맞는 img 파일 path 정보가 들어있는 파일
# ###############################################################################################

import os
import shutil
from sklearn.model_selection import train_test_split

# 원본 라벨 경로
label_dir = "data/train_labels"

# 이미지 파일이 있는 곳
image_dir = "data/train_images"

# 분할된 라벨 저장할 경로
train_label_dir = "data/train_labels/train"
val_label_dir = "data/train_labels/val"

# train.txt / val.txt 저장 경로
train_txt_path = "data/train.txt"
val_txt_path = "data/val.txt"

# 폴더 생성
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 원본 라벨 리스트 수집
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
image_files = [f.replace(".txt", ".png") for f in label_files]

# TODO: stratify
class_ids = []
try:
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
                    
except ValueError:
    print(f"오류: 잘못된 형식의 데이터 - {label_dir}")
except Exception as e:
    print(f"오류: 파일 처리 중 예외 발생 - {e}")
    
labels_for_stratify = list(class_ids)

# 분할
train_labels, val_labels = train_test_split(label_files, test_size=0.2, random_state=42, stratify=labels_for_stratify)

# .txt 파일 초기화
open(train_txt_path, "w").close()
open(val_txt_path, "w").close()

# 복사 및 경로 저장a
for label_file in train_labels:
    shutil.move(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))
    img_path = os.path.join(image_dir, label_file.replace(".txt", ".png"))
    with open(train_txt_path, "a") as f:
        f.write(f"{img_path}\n")

for label_file in val_labels:
    shutil.move(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))
    img_path = os.path.join(image_dir, label_file.replace(".txt", ".png"))
    with open(val_txt_path, "a") as f:
        f.write(f"{img_path}\n")

print("train_labels → train_labels/train & val 분할 완료")
print("train.txt / val.txt 파일 생성 완료")

################ YOLO 모델 학습을 위한 data.yaml 파일을 생성하는 함수. ################
# TODO: startify 를 이용해서 train, val 나누기
# TODO: yaml 파일에 노클래스랑, 백그라운드 없애야합니다.
import os 
import yaml
from src.data_utils.data_loader import get_category_mapping

def make_yaml_file(YOLO_dataset_name='dataset_1', output_dir=os.path.join("data"), ):
    """
    YOLO 모델 학습을 위한 data.yaml 파일을 생성하는 함수.
    이미지폴더를 분리하지 않고 yaml 파일을 생성합니다.
    train:과 val:이 같은 폴더를 가리키므로, YOLO가 _val.txt, _train.txt을 참조해 자동으로 이미지를 찾을 수 있습니다. 
    args:
        YOLO_dataset_name (str): 생성할 데이터셋 이름 (yaml 파일이름으로 만들기 위한 정보)
        output_path (str): 생성될 YAML 파일 경로 (기본값: dataset.yaml)
    """
    # YAML 파일 경로 설정
    yaml_file_dir = os.path.join(output_dir, f"{YOLO_dataset_name}.yaml")
    
    # 클래스와 클래스 수 설정
    _, idx_to_name = get_category_mapping("data/train_annots_modify") 
    keys = list(idx_to_name.keys())  
    del idx_to_name[keys[0]], idx_to_name[keys[-1]]   # 마지막 키: ㄴno_class, 첫번째 키: ㄴbackground 삭제
    class_names = [name for name in idx_to_name.values()]

    #  make_yaml_file 함수에서 사용할 txt 파일을 생성
    image_folder ='data/train_images'
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png'))]

    train_files, val_files = [], []
    for label_file in train_labels:
        train_files.append(os.path.join(train_label_dir, label_file))
        val_files.append(os.path.join(val_label_dir, label_file))

    # 학습 데이터셋 파일 경로 저장
    with open(f"data/train_labels/{YOLO_dataset_name}.txt", "w") as f:
        for file in train_files:
            f.write(file + '\n')

    # 검증 데이터셋 파일 경로 저장
    with open(f"data/val_labels/{YOLO_dataset_name}.txt", "w") as f:
        for file in val_files:
            f.write(file + '\n')

    # train:과 val:이 같은 폴더를 가리키므로, YOLO가 _val.txt, _train.txt을 참조해 자동으로 이미지를 찾을 수 있습니다. 
    data = {
        "train": f"train.txt",  
        "val": f"val.txt", 
        "train_labels":f"train_labels/{YOLO_dataset_name}.txt",  
        "val_labels":f"val_labels/{YOLO_dataset_name}.txt",   
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }

    try:
        with open(output_dir, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"YAML 파일이 '{output_dir}'에 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"YAML 파일 생성 중 오류가 발생했습니다: {e}")

    print(f"{yaml_file_dir} 파일이 생성되었습니다.")

###########################################################################################


if __name__ == "__main__":  # 모듈화 전 test code
    make_yaml_file(YOLO_dataset_name='dataset_1', output_dir=os.path.join("data", "dataset_1.yaml"))
    print("data.yaml 파일 생성 완료")