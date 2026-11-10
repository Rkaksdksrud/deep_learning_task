# Task 1 v1 — 학습된 단일 DNN

이 폴더만 받아도 재학습 없이 추론할 수 있습니다. `task1.pt`는 모델 가중치뿐 아니라
전처리 통계, 입력 열 순서, 모델 구조를 함께 포함합니다. 전체 저장소나 학습 데이터는 필요 없습니다.

## 사용

압축을 푼 폴더에서 Python 3.13으로 다음 명령을 실행합니다.

```powershell
python -m pip install -r requirements.txt
python predict.py --data "input.csv" --output "predictions.csv"
```

`input.csv`는 사용자가 준비한 예측 대상 데이터입니다. `feature_names.json`에 적힌 23개 열이 필요합니다.
열 순서는 달라도 되며 `sample_id`는 선택 항목입니다. 정답 열은 필요 없습니다.
결과는 원래 행 순서의 `sample_id,probability`입니다.
확률은 24개월 안에 90일 이상 연체할 가능성을 나타내는 모델 출력입니다.
확률 보정이 별도로 검증된 모델은 아닙니다.

다른 Python 코드에서 사용:

```python
import pandas as pd
from predict import predict_proba
probabilities = predict_proba(pd.read_csv("input.csv"), "task1.pt")
```

## 모델 및 평가

- 자료: 과제에서 제공한 HELOC 7,896행. 외부 학습 데이터와 앙상블 사용 없음.
- 입력: 수치형 23개 + 특수값/결측 표시 및 결측 비율. Robust scaling.
- 구조: 완전연결 은닉층 32개, GELU, Dropout 0.1, 출력 logit 1개.
- 24개 설정 × 5-fold 중 선택. 평균 CV AUROC: 0.8053453481.
- 개발 데이터만으로 학습한 평가 모델의 holdout AUROC: 0.8139440453 (1,580행).
- 배포 가중치는 모든 제공 데이터를 40 epochs 재학습한 단일 모델입니다.
- 위 holdout 점수는 전체 재학습 모델에 대한 독립 평가가 아닙니다. private Test AUROC는 미확인입니다.
- 원본 기록 날짜: 2026-11-10. 교육용 과제 결과이며 실제 신용 의사결정용으로 검증되지 않았습니다.

모델 파일의 재배포본은 로컬 데이터 경로만 상대경로로 정리했습니다. 가중치와 전처리는 동일합니다.
`SHA256SUMS.json`으로 각 파일의 무결성을 확인할 수 있습니다.
