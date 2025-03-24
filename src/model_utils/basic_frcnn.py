import torch
from tqdm import tqdm
import os
from src.data_utils.data_loader import get_loader, get_category_mapping
from utils import get_optimizer, get_scheduler  # utils.py에서 가져오기
from src.model_utils.basic_YOLO import get_yolov5  # YOLO 모델
from ultralytics.utils.loss import ComputeLoss


def train_YOLO(img_dir, ann_dir, batch_size=8, num_epochs=5, lr=0.001, weight_decay=0.005, optimizer_name="sgd", scheduler_name="step", device="cpu", debug=False):
    """
    YOLOv5 모델을 학습시키는 함수
    - 데이터 로더 설정
    - YOLO 모델 초기화
    - 학습 루프 (훈련 및 검증)
    - 모델 저장
    """
    
    # ✅ 데이터 로더 생성
    train_loader = get_loader(img_dir, ann_dir, batch_size, mode="train", val_ratio=0.2, debug=debug)
    val_loader = get_loader(img_dir, ann_dir, batch_size, mode="val", val_ratio=0.2, debug=debug)

    # ✅ 카테고리 매핑 가져오기
    name_to_idx, idx_to_name = get_category_mapping(ann_dir)
    num_classes = len(name_to_idx)  # 클래스 개수 설정

    # ✅ YOLOv5 모델 로드
    model = get_yolov5(model_path="yolov5s.pt", num_classes=num_classes).to(device)

    # ✅ 옵티마이저, 스케줄러, 손실 함수 정의
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler_name, optimizer, T_max=100)
    compute_loss = ComputeLoss(model)

    best_val_loss = float("inf")  # 최적 검증 손실값 저장

    # ✅ 학습 루프
    for epoch in range(num_epochs):
        print(f"\n🚀 Epoch {epoch+1}/{num_epochs} 시작...")

        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, total=len(train_loader), desc=f"🟢 Training {epoch+1}/{num_epochs}")
        for imgs, targets in train_bar:
            imgs = imgs.to(device)

            # ✅ 타겟 데이터 YOLO 형식 변환
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # ✅ 모델 예측
            preds = model(imgs)

            # ✅ YOLOv5의 ComputeLoss를 사용하여 손실 계산
            loss, loss_items = compute_loss(preds, targets)

            # ✅ 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 그래디언트 클리핑
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())  # 진행 상태 표시

        # ✅ 스케줄러 업데이트
        if scheduler_name == "plateau":
            scheduler.step(total_loss)  # ReduceLROnPlateau는 loss를 인자로 받음
        else:
            scheduler.step()

        # ✅ 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f"🔵 Validation {epoch+1}/{num_epochs}")
            for imgs, targets in val_bar:
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                preds = model(imgs)
                loss, _ = compute_loss(preds, targets)
                val_loss += loss.item()
                val_bar.set_postfix(val_loss=loss.item())

        print(f"📉 Epoch {epoch+1} - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

        # ✅ 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ 검증 손실 개선됨! 모델 저장 중... (Best Val Loss: {best_val_loss:.4f})")
            save_model(model, epoch, best_val_loss)  # 모델 저장


def save_model(model, epoch, val_loss):
    """ 모델 가중치 저장 함수 """
    save_dir = "models/weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"yolov5_epoch_{epoch}_val_{val_loss:.4f}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"✅ 모델 저장 완료: {save_path}")


if __name__ == "__main__":
    train_YOLO(img_dir="data/train_images", ann_dir="data/train_annots_modify", device="cuda" if torch.cuda.is_available() else "cpu")
