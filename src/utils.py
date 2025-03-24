import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score

# 옵티마이저 생성 함수
def get_optimizer(name, model, lr=1e-3, weight_decay=0):
    """
    주어진 이름(name)에 해당하는 옵티마이저를 생성하여 반환합니다.

    :param name: 사용할 옵티마이저 이름 (예: 'adam', 'sgd', 'adamw')
    :param model: 학습할 모델의 파라미터
    :param lr: 학습률
    :param weight_decay: 가중치 감쇠 (L2 정규화)
    :return: 선택된 옵티마이저 인스턴스
    """
    optimizers = {
        "sgd": optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
        "adam": optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "adamw": optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        "rmsprop": optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
    }

    if name.lower() not in optimizers:
        raise ValueError(f"지원되지 않는 옵티마이저: {name}. 사용 가능한 옵션: {list(optimizers.keys())}")

    return optimizers[name.lower()]

# 스케줄러 생성 함수
def get_scheduler(name, optimizer, step_size=5, gamma=0.1, T_max=50):
    """
    주어진 이름(name)에 해당하는 스케줄러를 생성하여 반환합니다.

    :param name: 사용할 스케줄러 이름 (예: 'step', 'cosine', 'plateau')
    :param optimizer: 옵티마이저 인스턴스
    :param step_size: StepLR에서 사용하는 스텝 크기
    :param gamma: StepLR, ExponentialLR에서 사용되는 감쇠 계수
    :param T_max: CosineAnnealingLR에서 사용하는 주기 길이
    :return: 선택된 스케줄러 인스턴스
    """
    schedulers = {
        "step": lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max),
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=gamma, patience=step_size),
        "exponential": lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
    }

    if name.lower() not in schedulers:
        raise ValueError(f"지원되지 않는 스케줄러: {name}. 사용 가능한 옵션: {list(schedulers.keys())}")

    return schedulers[name.lower()]

# IoU 계산 함수
def calculate_iou(box1, box2):
    # box = [x_min, y_min, x_max, y_max]
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # 교집합 부분
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    # 교집합 영역이 존재하는지 확인
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0.0  # 교집합이 없으면 IoU는 0
    
    # 교집합 면적
    inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    # 합집합 면적
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # IoU 계산
    iou = inter_area / (area1 + area2 - inter_area)
    return iou

# AP 계산 함수
def calculate_ap(predictions, targets, iou_threshold=0.5):
    tp = []  # True Positive
    fp = []  # False Positive
    fn = []  # False Negative
    matched_targets = []  # 타겟 매칭 추적

    # 예측을 confidence에 따라 내림차순 정렬
    predictions = sorted(predictions, key=lambda x: max(x['scores']), reverse=True)

    for pred in predictions:
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        
        for pred_bbox, pred_class_id in zip(pred_boxes, pred_labels):
            target_bboxes = [target['boxes'] for target in targets if pred_class_id in target['labels']]
            
            if not target_bboxes:
                fp.append(1)
                continue
            
            target_bboxes = [bbox for sublist in target_bboxes for bbox in sublist]

            ious = [calculate_iou(pred_bbox, target_bbox) for target_bbox in target_bboxes]
            
            matched = False
            for iou, target_bbox in zip(ious, target_bboxes):
                if iou >= iou_threshold and target_bbox not in matched_targets:
                    tp.append(1)
                    fp.append(0)
                    matched_targets.append(target_bbox)
                    matched = True
                    break
            
            if not matched:
                tp.append(0)
                fp.append(1)
    
    tp = np.array(tp)
    fp = np.array(fp)

    # 길이가 맞지 않다면, 길이가 맞게 패딩을 추가하거나 누락된 값을 채워주는 방법을 사용
    if len(tp) != len(fp):
        min_len = min(len(tp), len(fp))
        tp = tp[:min_len]
        fp = fp[:min_len]
    
    cumsum_tp = np.cumsum(tp)  # 누적 TP
    cumsum_fp = np.cumsum(fp)  # 누적 FP

    precision = np.divide(cumsum_tp, (cumsum_tp + cumsum_fp), where=(cumsum_tp + cumsum_fp) != 0)
    recall = cumsum_tp / len(targets)  # Recall
    
    # AP 계산 시 Precision-Recall 배열 길이가 맞는지 확인
    if len(precision) != len(recall):
        min_len = min(len(precision), len(recall))
        precision = precision[:min_len]
        recall = recall[:min_len]
    
    ap = np.trapz(precision, recall)  # 면적 계산
    return ap

# def calculate_ap(predictions, targets, iou_threshold=0.5):
#     # predictions: 예측된 값 [{'boxes': [...], 'labels': [...], 'score': [...]}]
#     # targets: 실제 값 [{'boxes': BoundingBoxes([...]), 'labels': tensor([...])}]
    
#     tp = []  # True Positive
#     fp = []  # False Positive
#     fn = []  # False Negative
    
#     # 예측을 confidence에 따라 내림차순 정렬
#     predictions = sorted(predictions, key=lambda x: max(x['scores']), reverse=True)
    
#     # 클래스별로 처리
#     for pred in predictions:
#         pred_boxes = pred['boxes']
#         pred_labels = pred['labels']
        
#         matched_targets = []
#         for pred_bbox, pred_class_id in zip(pred_boxes, pred_labels):
#             target_bboxes = [target['boxes'] for target in targets if pred_class_id in target['labels']]
            
#             # 타겟 박스가 없으면 False Positive 처리
#             if not target_bboxes:
#                 fp.append(1)
#                 continue

#              # 2차원 리스트 → 1차원 리스트 변환
#             target_bboxes = [bbox for sublist in target_bboxes for bbox in sublist]

#             # IoU 계산하여 가장 높은 값 찾기
#             ious = [calculate_iou(pred_bbox, target_bbox) for target_bbox in target_bboxes]
            
#             # # IoU가 threshold 이상이면 True Positive
#             # if any(iou >= iou_threshold for iou in ious):
#             #     tp.append(1)
#             #     fp.append(0)
#             # else:
#             #     tp.append(0)
#             #     fp.append(1)
#             matched = False
#             for iou, target_bbox in zip(ious, target_bboxes):
#                 if iou >= iou_threshold and target_bbox not in matched_targets:
#                     tp.append(1)
#                     fp.append(0)
#                     matched_targets.append(target_bbox)
#                     matched = True
#                     break
            
#             if not matched:
#                 tp.append(0)
#                 fp.append(1)

    
#     # Precision과 Recall 계산
#     tp = np.array(tp)
#     fp = np.array(fp)

#     # 길이가 맞지 않다면, 길이가 맞게 패딩을 추가하거나 누락된 값을 채워주는 방법을 사용
#     if len(tp) != len(fp):
#         min_len = min(len(tp), len(fp))
#         tp = tp[:min_len]
#         fp = fp[:min_len]
    
#     cumsum_tp = np.cumsum(tp)  # 누적 TP
#     cumsum_fp = np.cumsum(fp)  # 누적 FP
    
#     # precision = cumsum_tp / (cumsum_tp + cumsum_fp)  # Precision
#     precision = np.divide(cumsum_tp, (cumsum_tp + cumsum_fp), where=(cumsum_tp + cumsum_fp) != 0)
#     recall = cumsum_tp / len(targets)  # Recall

#     # AP 계산 시 Precision-Recall 배열 길이가 맞는지 확인
#     if len(precision) != len(recall):
#         min_len = min(len(precision), len(recall))
#         precision = precision[:min_len]
#         recall = recall[:min_len]
    
#     # AP 계산 (Precision-Recall Curve의 면적)
#     ap = np.trapz(precision, recall)  # 면적 계산 (곡선 아래 면적)
    
#     return ap

# mAP 계산 함수
def calculate_map(predictions, targets, num_classes, iou_threshold=0.5):
    ap_values = []
    
    for class_id in range(1, num_classes+1):  # 클래스 ID는 1부터 시작한다고 가정
        # 예측값과 실제값을 클래스별로 필터링
        class_predictions = [p for p in predictions if class_id in p['labels']]
        class_targets = [t for t in targets if class_id in t['labels']]
        
        # 해당 클래스에 대해 AP 계산
        precision, recall, ap = calculate_ap(class_predictions, class_targets, iou_threshold)
        ap_values.append(ap)
    
    # mAP는 모든 클래스의 AP의 평균
    map_score = np.mean(ap_values)
    return map_score, precision, recall

# 시각화 함수
def draw_bbox(ax, box, text, color):
    """
    - ax: matplotlib Axes 객체
    - box: 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
    - text: 바운딩 박스 위에 표시할 텍스트
    - color: 바운딩 박스와 텍스트의 색상
    """
    ax.add_patch(
        plt.Rectangle(
            xy=(box[0], box[1]),
            width=box[2] - box[0],
            height=box[3] - box[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
    )
    ax.annotate(
        text=text,
        xy=(box[0] - 5, box[1] - 5),
        color=color,
        weight="bold",
        fontsize=13,
    )

# f1 스코어 계산 함수 
def f1_score(precision, recall):
    score = 2 * ((precision * recall)/ (precision + recall))

    return score

import numpy as np
def visualization(results, page_size=20, page=0, debug=True):
    total_num = len(results)
    start_idx = page * page_size
    end_idx = min(start_idx + page_size, total_num)

    print(f"페이지 {page + 1} / {np.ceil(total_num / page_size).astype(int)} | {start_idx} - {end_idx}번째 이미지 표시")

    num_images = min(total_num, 20)
    row_img = max(num_images // 4, 1)
    col_img = max(num_images // row_img, 1)
    figsize = (5 * col_img, 5 * row_img)

    print(f"페이지 당 {row_img} 행, {col_img} 열 형태의 이미지 플롯")

    fig, ax = plt.subplots(row_img, col_img, figsize=figsize)

    if row_img == 1 or col_img == 1:
        ax = np.expand_dims(ax, axis=0)  
        
    for i in range(start_idx, end_idx):
        file_name = results[i]['file_name']
        image_id = results[i]['category_id']
        boxes = results[i]['boxes']
        bbox_num = len(boxes)
        path = os.path.join('../data/test_images', file_name)
        dr_name = [idx_to_name[id] for id in image_id]

        if debug:
            print(f"[{i + 1}] Visualize Image: {file_name}, DRUG ID: {image_id}, BBox Num: {bbox_num}")


        if not os.path.exists(path):
            print(f"[Error] 이미지 경로를 찾을 수 없습니다: {path}")
            continue 

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax_idx = i - start_idx
        ax_row = ax_idx // col_img
        ax_col = ax_idx % col_img

        ax[ax_row, ax_col].imshow(image)

        for j in range(bbox_num):
            ax[ax_row, ax_col].add_patch(
                plt.Rectangle(
                    (boxes[j][0], boxes[j][1]),
                    boxes[j][2] - boxes[j][0],
                    boxes[j][3] - boxes[j][1],
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
            )
            ax[ax_row, ax_col].annotate(
                text=dr_name[j],
                xy=(boxes[j][0] - 10, boxes[j][1] - 10),
                color="red",
                weight="bold",
                fontsize=8
            )

        ax[ax_row, ax_col].axis("off")
        ax[ax_row, ax_col].set_title(f"{file_name}")

    plt.tight_layout()
    plt.show()
