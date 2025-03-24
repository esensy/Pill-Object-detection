import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score

# 시각화에 쓸 알약 이름
"""
{0: '보령부스파정 5mg',
 1: '뮤테란캡슐 100mg',
 2: '일양하이트린정 2mg',
 3: '기넥신에프정(은행엽엑스)(수출용)',
 4: '무코스타정(레바미피드)(비매품)',
 5: '알드린정', 6: '뉴로메드정(옥시라세탐)',
 7: '타이레놀정500mg', 8: '에어탈정(아세클로페낙)',
 9: '삼남건조수산화알루미늄겔정',
 10: '타이레놀이알서방정(아세트아미노펜)(수출용)',
 11: '삐콤씨에프정 618.6mg/병',
 12: '조인스정 200mg',
 13: '쎄로켈정 100mg',
 14: '넥시움정 40mg',
 15: '리렉스펜정 300mg/PTP',
 16: '아빌리파이정 10mg', 
 17: '자이프렉사정 2.5mg', 
 18: '다보타민큐정 10mg/병', 
 19: '써스펜8시간이알서방정 650mg', 
 20: '에빅사정(메만틴염산염)(비매품)', 
 21: '리피토정 20mg', 
 22: '크레스토정 20mg', 
 23: '가바토파정 100mg', 
 24: '동아가바펜틴정 800mg', 
 25: '오마코연질캡슐(오메가-3-산에틸에스테르90)', 
 26: '란스톤엘에프디티정 30mg', 
 27: '리리카캡슐 150mg', 
 28: '종근당글리아티린연질캡슐(콜린알포세레이트)\xa0', 
 29: '콜리네이트연질캡슐 400mg', 
 30: '트루비타정 60mg/병', 
 31: '스토가정 10mg', 
 32: '노바스크정 5mg', 
 33: '마도파정', 
 34: '플라빅스정 75mg', 
 35: '엑스포지정 5/160mg',
 36: '펠루비정(펠루비프로펜)',
 37: '아토르바정 10mg',
 38: '라비에트정 20mg',
 39: '리피로우정 20mg',
 40: '자누비아정 50mg',
 41: '맥시부펜이알정 300mg',
 42: '메가파워정 90mg/병',
 43: '쿠에타핀정 25mg',
 44: '비타비백정 100mg/병',
 45: '놀텍정 10mg',
 46: '자누메트정 50/850mg', 
 47: '큐시드정 31.5mg/PTP',
 48: '아모잘탄정 5/100mg',
 49: '세비카정 10/40mg',
 50: '트윈스타정 40/5mg',
 51: '카나브정 60mg',
 52: '울트라셋이알서방정',
 53: '졸로푸트정 100mg',
 54: '트라젠타정(리나글립틴)',
 55: '비모보정 500/20mg',
 56: '레일라정',
 57: '리바로정 4mg',
 58: '렉사프로정 15mg',
 59: '트라젠타듀오정 2.5/850mg',
 60: '낙소졸정 500/20mg',
 61: '아질렉트정(라사길린메실산염)',
 62: '자누메트엑스알서방정 100/1000mg',
 63: '글리아타민연질캡슐',
 64: '신바로정',
 65: '에스원엠프정 20mg', 
 66: '브린텔릭스정 20mg',
 67: '글리틴정(콜린알포세레이트)',
 68: '제미메트서방정 50/1000mg',
 69: '아토젯정 10/40mg',
 70: '로수젯정10/5밀리그램',
 71: '로수바미브정 10/20mg',
 72: '카발린캡슐 25mg',
 73: '케이캡정 50mg'} 
"""

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
def get_scheduler(name, optimizer, step_size=10, gamma=0.1, T_max=50):
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
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=step_size),
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
    # predictions: 예측된 값 [{'boxes': [...], 'labels': [...], 'score': [...]}]
    # targets: 실제 값 [{'boxes': BoundingBoxes([...]), 'labels': tensor([...])}]
    
    tp = []  # True Positive
    fp = []  # False Positive
    fn = []  # False Negative
    
    # 예측을 confidence에 따라 내림차순 정렬
    predictions = sorted(predictions, key=lambda x: max(x['scores']), reverse=True)
    
    # 클래스별로 처리
    for pred in predictions:
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        
        for pred_bbox, pred_class_id in zip(pred_boxes, pred_labels):
            target_bboxes = [target['boxes'] for target in targets if pred_class_id in target['labels']]

            # 클래스에 해당하는 target_bboxes가 없을 경우 건너뛰기
            if not target_bboxes:
                continue  
            
            # BoundingBoxes에서 좌표 추출
            target_bboxes = target_bboxes[0].tolist()  # tensor를 리스트로 변환
            
            # IoU 계산하여 가장 높은 값 찾기
            ious = [calculate_iou(pred_bbox, target_bbox) for target_bbox in target_bboxes]
            
            # IoU가 threshold 이상이면 True Positive
            if any(iou >= iou_threshold for iou in ious):
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
    
    # Precision과 Recall 계산
    tp = np.array(tp)
    fp = np.array(fp)
    
    cumsum_tp = np.cumsum(tp)  # 누적 TP
    cumsum_fp = np.cumsum(fp)  # 누적 FP
    
    precision = cumsum_tp / (cumsum_tp + cumsum_fp)  # Precision
    recall = cumsum_tp / len(targets)  # Recall
    
    # AP 계산 (Precision-Recall Curve의 면적)
    ap = np.trapz(precision, recall)  # 면적 계산 (곡선 아래 면적)
    
    return ap

# mAP 계산 함수
def calculate_map(predictions, targets, num_classes, iou_threshold=0.5):
    ap_values = []
    
    for class_id in range(1, num_classes+1):  # 클래스 ID는 1부터 시작한다고 가정
        # 예측값과 실제값을 클래스별로 필터링
        class_predictions = [p for p in predictions if class_id in p['labels']]
        class_targets = [t for t in targets if class_id in t['labels']]
        
        # 해당 클래스에 대해 AP 계산
        ap = calculate_ap(class_predictions, class_targets, iou_threshold)
        ap_values.append(ap)
    
    # mAP는 모든 클래스의 AP의 평균
    map_score = np.mean(ap_values)
    return map_score

# 시각화 함수수
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
