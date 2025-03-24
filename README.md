# Develop Branch

진행상황을 업데이트 한다.


## 구극모:
### Test_yolo_temp.py 수정안

    parameter수에 변동사항이 있습니다.

    :param model: 학습된 YOLO 모델
    :param test_image_path: 테스트 이미지 경로
    :param model_path: 모델 가중치 경로 (선택)
    :param categories: 클래스 정보 (dict)
    :param debug: 디버그 모드 활성화 (True/False)
    :param save_results: 예측된 이미지를 저장할지 여부 (True/False)

    :return: 예측 결과 리스트

YOLO 모델에서의 예측된 결과는 시각적으로 보편화되어 있으나 객체개입에 있어서 오류가 자주 발생하는 것을 확인했습니다. 이 오류를 줄이고자 각 객체 탐색에 있어 isinstance를 도입하여 확인하는 부분을 추가했으며, 인풋으로 x_center,y_center, width, height형태로 받더라도 결과는 XYXY형태로 출력되는것을 확인. 이를 반영하여 최종 출력의 형태를 submission의 형태로 확인하기 위한 코드를 아래에 첨부했습니다. 이는 추가적인 토의를 통해, test 모듈에 들어갈지 아니면 make_csv의 조건부로 달릴지를 결정하는게 옳다고 판단됩니다.


### 추후 변경 사항:

Dataset, DataLoader의 적용을 고려하지 않고 test_image를 바로 받을 경우를 실행하는 모듈이므로 자동화 과정에서 Image를 바로 입력받을지 아니면 최소한의 이미지 증강과정 (Resize, toTenser 등)을 거치는게 나을지에 관해 토의 후 추가 수정을 고려.

현재는 모델을 입력받는 형식의 Method이나, 다른 모듈의 형태를 고려했을 때, 모듈 자체에서 model을 생성하고 path인자를 받는 방법으로 바꾸는 것이 일관성을 위해 고려되야하는 사항.
