from ultralytics import YOLO
import torch
import pandas as pd
import os
import argparse

def enable_weights_only_false():
    original_load = torch.load
    def custom_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = custom_torch_load
    print("[INFO] torch.load monkey-patched: weights_only=False")

def predict_yolo_and_export_csv(model_path, image_dir, conf_threshold=0.25, save_csv_path=None, verbose=False):
    """
    YOLO 모델을 테스트 이미지에 적용하고 예측 결과를 주어진 형식의 CSV로 저장하는 함수.

    Args:
        model_path (str): 학습된 YOLO 모델 가중치 경로 (예: 'best.pt')
        image_dir (str): 테스트 이미지 디렉토리 경로
        conf_threshold (float): confidence threshold (기본 0.25)
        save_csv_path (str or None): 결과 CSV 저장 경로 (None이면 자동 지정)
        verbose (bool): YOLO 내부 출력 여부

    Returns:
        pd.DataFrame: 예측 결과 DataFrame (annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score)
    """
    model = YOLO(model_path)
    results = model.predict(source=image_dir, save=False, conf=conf_threshold, verbose=verbose)

    if not results:
        raise RuntimeError("예측 결과가 비어 있습니다.")

    if save_csv_path is None:
        save_dir = results[0].save_dir or os.path.dirname(model_path) or os.getcwd()
        save_csv_path = os.path.join(save_dir, "submission.csv")

    submission = []
    annotation_id = 1
    for res in results:
        image_name = os.path.basename(res.path)
        image_id = int(os.path.splitext(image_name)[0])
        boxes = res.boxes.xywh.cpu().numpy()
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        for cls, box, score in zip(class_ids, boxes, scores):
            submission.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox_x": box[0],
                "bbox_y": box[1],
                "bbox_w": box[2],
                "bbox_h": box[3],
                "score": score
            })
            annotation_id += 1

    df = pd.DataFrame(submission)
    df.to_csv(save_csv_path, index=False)
    print(f"[INFO] 결과가 저장되었습니다: {save_csv_path}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="YOLO 가중치 파일 경로")
    parser.add_argument("--image_dir", type=str, required=True, help="테스트 이미지 디렉토리")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="YOLO 내부 출력 활성화 여부")
    parser.add_argument("--force_load", action="store_true", help="weights_only=False 강제 적용")
    args = parser.parse_args()

    if args.force_load:
        enable_weights_only_false()

    predict_yolo_and_export_csv(
        model_path=args.model_path,
        image_dir=args.image_dir,
        conf_threshold=args.conf_threshold,
        save_csv_path=None,
        verbose=args.verbose
    )