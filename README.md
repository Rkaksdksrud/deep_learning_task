# Task 1 실행

이 폴더에 전용 Python 환경 `.venv`, 필요한 라이브러리와 실제 데이터가 준비되어 있습니다.
VS Code 실행 버튼을 누르거나 인수 없이 실행하면 사용법이 표시됩니다.
다른 Python으로 `first_version.py`를 실행해도 이 폴더의 `.venv`로 자동 전환합니다.

현재 폴더의 PowerShell에서 본 학습:

```powershell
python first_version.py train
```

기본은 24개 설정 × 5-fold, 최대 200 epoch입니다. 결과는 `runs/task1`에 저장됩니다.
다시 실행할 때는 `--output runs/second`처럼 새로운 결과 폴더를 지정하세요.

빠른 동작 확인:

```powershell
python first_version.py train --output runs/quick --trials 2 --folds 2 --epochs 5 --patience 2
```

데이터 경로를 생략하면 이 폴더의 `data/heloc_release.csv`를 사용합니다.
모델은 단일 DNN이며 전처리, 파생비율, 하이퍼파라미터를 CV AUROC로 비교합니다.
holdout은 설정 선택 이후 한 번 평가하고, 최종 모델은 전체 데이터로 재학습합니다.
제출용 모델과 전처리가 함께 `task1.pt`에 저장됩니다.

예측:

```powershell
python first_version.py predict --data data/test.csv --weights runs/task1/task1.pt --output predictions.csv
```

환경 복원: Python 환경을 다시 만든 뒤 `python -m pip install -r requirements.txt`를 실행하세요.
원래 코드는 `first_version.py.before-env-fix.bak`에 보존했습니다.


## GitHub에서 모델만 받기

학습이 끝난 v1 모델은 `models/task1-v1/`에 고정 보관합니다. 재학습 없이 사용하려면
[모델 전용 ZIP 다운로드](https://github.com/ssung060908-prog/deep_learning_task/raw/refs/heads/main/models/task1-v1.zip)를 받아 압축을 풀고 안의 README를 따르세요.

- `task1.pt`: 가중치 + 전처리 + 모델 구조
- `predict.py`: 독립 실행 가능한 예측 코드
- `requirements.txt`: 라이브러리 버전
- `feature_names.json`: 필요한 입력 열
- `metrics.json`: 학습·평가 요약
- `SHA256SUMS.json`: 파일 해시

최초 고정 버전의 Git 태그는 `task1-v1`입니다. 이후 main이 바뀌어도
[고정 버전 모델 ZIP](https://github.com/ssung060908-prog/deep_learning_task/raw/refs/tags/task1-v1/models/task1-v1.zip)을 사용할 수 있습니다.

현재 모델: 은닉층 [32], GELU, Robust scaling. CV AUROC 0.80535, holdout AUROC 0.81394.
가중치는 전체 데이터 40 epochs 재학습 결과이며 holdout 점수는 개발 데이터만으로 학습한 평가 모델의 결과입니다.
private Test 점수는 아직 모릅니다.

`experiments/task1-v1/`에는 실험 요약, `docs/model_design_roadmap.md`에는 후속 설계 계획이 있습니다.
원본 과제 데이터는 공유하지 않습니다. 새 컴퓨터에서 재학습하려면 허가받은 데이터를 별도로
`data/heloc_release.csv`에 준비하고 필요한 라이브러리를 설치해야 합니다.
이 저장소는 Task 1 결과이며 Task 2와 보고서를 포함한 전체 과제 제출물은 아닙니다.
