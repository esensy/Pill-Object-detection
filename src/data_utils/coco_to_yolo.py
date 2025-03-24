################################################################################################
# 데이터 다운 -> data loader 실행 -> json modify 실행 -> coco to yolo 순서
# 실행 코드
# python coco_to_yolo.py --json_folder data/train_annots_modify --output_dir data/train_labels_YOLO
# category_id x_center y_center width height + 좌표 정규화
# 이렇게 바꿔놓아야 YOLO에서 돌아간다고 합니다
# ###############################################################################################

import json
import os
import argparse
from src.data_utils.data_loader import get_category_mapping

def convert_coco_to_yolo(json_file, output_dir):
    """
    COCO JSON 형식의 어노테이션 데이터를 YOLO 형식으로 변환하는 함수.

    Args:
        json_file (str): COCO 형식의 JSON 어노테이션 파일 경로
        output_dir (str): YOLO 형식의 라벨 파일이 저장될 디렉토리

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    ANN_DIR = "data/train_annots_modify"
    name_to_idx, _ = get_category_mapping(ANN_DIR)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img in data["images"]:
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        label_path = os.path.join(output_dir, f"{img['file_name'].replace('.png', '.txt')}")

        with open(label_path, "w", encoding="utf-8") as f:
            for ann in data["annotations"]:
                if ann["image_id"] == img_id:
                    x, y, w, h = ann["bbox"]
                    x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
                    w, h = w / img_w, h / img_h

                    for category in data['categories']:
                        if ann["category_id"] == category["id"]:
                            category_id = name_to_idx[category['name']]
                            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def process_all_json(json_folder, output_dir):
    """
    폴더 내 모든 COCO JSON 파일을 YOLO 형식으로 변환하는 함수.

    Args:
        json_folder (str): 변환할 COCO JSON 파일이 저장된 폴더
        output_dir (str): YOLO 라벨을 저장할 폴더
    """
    if not os.path.exists(json_folder):
        print(f"❌ JSON 폴더가 존재하지 않습니다: {json_folder}")
        return

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if len(json_files) == 0:
        print("❌ 변환할 JSON 파일이 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 변환 진행
    for i, json_file in enumerate(json_files, start=1):
        json_path = os.path.join(json_folder, json_file)
        convert_coco_to_yolo(json_path, output_dir)


    print("🎉 모든 JSON 파일 변환 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format")
    parser.add_argument("--json_folder", type=str, default="data/train_annots_modify", required=True, help="Folder containing COCO JSON files")
    parser.add_argument("--output_dir", type=str, default="data/train_labels_YOLO", required=True, help="Output directory for YOLO label files")
    args = parser.parse_args()

    process_all_json(args.json_folder, args.output_dir)
