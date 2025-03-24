import torch
from yolov5.models.yolo import Model  # YOLOv5 모델 클래스


def get_yolov5(model_path=None, num_classes=80, device="cpu"):
    """
    YOLOv5 모델을 로드하고, 클래스 수에 맞게 출력 레이어를 수정합니다.

    Args:
        model_path (str, optional): 학습된 가중치 파일 경로 (e.g., yolov5s.pt). None일 경우 랜덤 초기화.
        num_classes (int): 클래스 개수
        device (str): 'cpu' 또는 'cuda'

    Returns:
        model (torch.nn.Module): YOLOv5 모델 객체
    """
    # ✅ YOLOv5s 기본 구조로 모델 초기화
    cfg_path = "yolov5/models/yolov5s.yaml"  # 기본 모델 구성 yaml
    model = Model(cfg=cfg_path, ch=3, nc=num_classes).to(device)

    # ✅ 하이퍼파라미터 수동 설정 (hyp.scratch.yaml 대체)
    hyp = {
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'fl_gamma': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    model.hyp = hyp
    model.model.hyp = hyp  # ComputeLoss에서 필요하므로 내부에도 설정

    # ✅ 학습된 weight 불러오기 (선택사항)
    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)
        print(f"✅ Pretrained weights loaded from {model_path}")

    return model
