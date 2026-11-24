# Deep Learning 팀 과제 · 진행 공유

각자의 실험은 별도 브랜치에서 진행하고, 코드·설정·집계 결과·버전별 모델 다운로드를 함께 남깁니다.
이번 Task2 결과는 `Rkaksdksrud/deep_learning_task`의 **`ktg_tried`** 브랜치에서 공유합니다.

| 구분 | 현재 상태 | 기록/사용법 |
|---|---|---|
| 기존 Task1 v1 | 단일 DNN 학습·배포 기록 보존 | [모델](models/task1-v1/README.md), [실험](experiments/task1-v1/README.md) |
| Task2 ResNet18 | 전체7개 run 학습·저장가중치 검증 완료 | [실험 비교](experiments/task2-resnet18-20261111/README.md), [학습 코드](task2/README.md) |
| Task2 후속 실험 | GoogLeNet·VGG11·ResNet320, 전체5개 run 학습·저장가중치 검증 완료 | [새 실험 비교](experiments/task2-followups-20261118/README.md) |
| Task2 학습 설정 비교 | 무증강 ResNet18 8개 full run·독립 검증 완료 | [설정 비교](experiments/task2-training-settings-20261124/README.md), [모델8개](models/task2-training-settings-20261124/README.md) |
| Task2 모든 full try의 가중치 | 기존7개 + 후속5개 + 설정 비교8개, 총20개 단일 모델 ZIP | [기존7개](models/task2-resnet18-20261111/README.md), [후속5개](models/task2-followups-20261118/README.md), [설정 비교8개](models/task2-training-settings-20261124/README.md) |

## Task2 결과

ResNet18의 학습 전용 수평반전 p=.5 + ±7° 회전 정책을 채택했습니다.
같은 조건 대조군의 BA는 **47.55→90.47%, 60.11→90.17%, 65.05→88.76%**였고,
세 조건 평균은 **57.57→89.80%(+32.23%p)**입니다.
같은 환자 fold를 두 seed에서 반복했으므로 독립3-fold나 private Test 점수가 아닙니다.
특정 검증 환자의 BA50%,26.25% 실패도 남아 있습니다.
모델은 모두 제공 데이터의 Train partition으로 학습한 단일 holdout 모델이며 전체 데이터 최종 제출 모델은 아닙니다.

11월18일에는 같은 환자 분할·seed42로 **5개 full run, 70epoch, 순수 학습43.77분**을 추가했습니다.

| 후속 시도 | Valid BA | 판단 |
|---|---:|---|
| GoogLeNet224, fold0 |90.0262%|지정 모델의 첫 full baseline 확보. 다른 fold 확인 필요|
| VGG11·224, lr3e-4, fold0 |50.0000%|전부 Absent로 예측하는 학습 실패|
| VGG11·224, lr1e-4, fold0 |80.2102%|학습 회복, 과적합과 환자별 실패 남음|
| 같은 VGG lr1e-4, fold1 |76.2810%|두 fold 레시피 평균78.2456%. 추가 seed 검증 없음|
| ResNet18·320, fold0 |89.5709%|224의90.4691%보다0.8982%p 하락, 시간1.47배여서 기각|

ResNet18·224를 참조 설정으로 유지합니다. 낮춘 VGG 학습률의 직접 대조는 fold0에서만 수행했으며,
위 두 fold 평균을 학습률 개선의 평균 효과로 해석하지 않습니다. GoogLeNet에는 자체 무증강 full 대조군이 없습니다.
모든 점수는 방법·checkpoint 선택에 사용한 개발용 validation 결과입니다.

## 11월24일 무증강 학습 설정 비교

사용자 요청에 따라 **반전과 회전을 모두 끄고**, 같은 ResNet18·환자 분할·seed에서 설정 하나씩만 바꿨습니다. 새 full run 8개를 GPU에서 순차 실행했고 실측 학습47.13분, 저장 모델 독립 검증8개 및 모든 실제 ZIP의 CPU 추론·체크섬 검증을 완료했습니다. 기존 실험을 대체하는 새 최고 모델을 채택하지 않았습니다.

| reference에서 바꾼 것 | fold0/seed42 BA | 같은 reference 대비 | 추가 확인·판단 |
|---|---:|---:|---|
| 없음: weighted CE·lr3e-4·WD1e-4·cosine |47.55%|—|기존 대조군 가중치·예측 재현 확인|
| 학습 weighted CE → 일반 CE |59.65%|+12.10%p|fold1 +13.23%p, 다른 seed -5.30%p. 사전 채택 gate 실패|
| lr3e-4 → 1e-4 |56.44%|+8.89%p|fold1 +6.24%p, 두 fold 평균 +7.56%p. 추가 seed 미확인|
| weight decay1e-4 → 0 |53.94%|+6.39%p|단일 조건 탐색, 최적 강도 미확인|
| cosine → constant LR |53.89%|+6.33%p|단일 조건 탐색, 일반적 우열 미확인|

학습률은 후속 확인 우선 후보입니다. 다만 fold0 양성 recall이19.76%이고, 일반 CE는 세 조건 평균 +6.68%p여도 seed에서 악화했습니다. 평균만으로 설정을 채택하지 않습니다. 설정별 비교 조건 수가 다르고 모든 점수는 개발용 validation입니다. 조합 효과·강도 최적값·증강 조건으로의 전이는 검증하지 않았습니다. 기존 기하 증강 ResNet18·224 참조 설정을 유지합니다.

[새 결과 기록](experiments/task2-training-settings-20261124/README.md)에서 설정·epoch·시간·혼동행렬·채택 제한을 확인할 수 있습니다. 현재 새8개 ZIP은 로컬 검증 완료 상태이며 Release 게시 확인 중입니다.

## 팀원이 이번 브랜치와 모델 받기

기존 작업 폴더와 분리해 내려받는 예시입니다.

```bash
git clone --branch ktg_tried --single-branch https://github.com/Rkaksdksrud/deep_learning_task.git deep_learning_task-ktg
cd deep_learning_task-ktg
python models/task2-resnet18-20261111/download.py --list
python models/task2-resnet18-20261111/download.py --run geometry_f0_s42 --output downloads/task2/geometry_f0_s42
```

다운로드한 ZIP에는 가중치·독립 추론 코드·환경·설정/지표·파일 SHA256이 함께 있습니다.
`download.py`는 ZIP과 가중치 해시를 확인합니다. Release가 게시되어 있어야 URL 다운로드가 가능합니다.
다른6개 try도 `--list`에 나온 ID로 같은 방법으로 받을 수 있습니다.

후속5개 모델은 별도 manifest와 Release를 사용합니다.

```bash
python models/task2-followups-20261118/download.py --list
python models/task2-followups-20261118/download.py --run googlenet_20261118_geometry_f0_s42 --output downloads/task2/googlenet-followup
python downloads/task2/googlenet-followup/predict.py --images input_images --output googlenet_predictions.csv --device cpu
```

학습 설정 비교8개는 세 번째 manifest와 Release를 사용합니다.

```bash
python models/task2-training-settings-20261124/download.py --list
python models/task2-training-settings-20261124/download.py --run resnet18_20261124_lr1e4_f1_s42 --output downloads/task2/lr1e4-f1
```

Task2는 Python3.10.13에서 확인했습니다. Task1 환경과 분리해 필요한 패키지를 설치한 뒤 추론합니다.

```bash
python -m pip install -r task2/requirements.txt
python downloads/task2/geometry_f0_s42/predict.py --images input_images --output predictions.csv --device cpu
```

가중치는 Git commit에 넣지 않고 [기존 Release](https://github.com/Rkaksdksrud/deep_learning_task/releases/tag/task2-resnet18-20261111-v1), [후속 Release](https://github.com/Rkaksdksrud/deep_learning_task/releases/tag/task2-followups-20261118-v1), [학습 설정 Release](https://github.com/Rkaksdksrud/deep_learning_task/releases/tag/task2-training-settings-20261124-v1)에 연결합니다. 다운로드 URL은 해당 Release가 게시되어 있어야 동작합니다.
원본 과제 데이터·영상/라벨·표본별 예측·분할표·영상이 포함된 노트북·과제 PDF는 이 공유본에 포함하지 않습니다.
재학습/검증에는 허가받은 데이터를 각자 `data/task2/`에 준비합니다.

## 앞으로 진행 상태를 공유하는 방식

1. 각자 작업 브랜치와 고유한 실험 ID를 사용합니다. 기존 실험 기록과 모델 버전은 덮어쓰지 않습니다.
2. `experiments/<실험ID>/`에 가설·config·seed/fold·지표·시간·결론과 코드 버전을 남깁니다.
3. `models/<모델묶음ID>/manifest.json`으로 각 run의 모델 ZIP·해시·지표를 연결합니다. 큰 파일은 Releases에 둡니다.
4. 모델은 하나씩 독립 평가합니다. 여러 try의 가중치를 공유한다는 것이 앙상블 사용을 뜻하지 않습니다.
5. 원본 데이터와 개별 표본 기록은 각자의 로컬에 보관합니다. 검증 점수와 전체 데이터 최종 모델 점수를 구분합니다.

현재 Task1 코드/모델/결과는 기존 내용을 유지했습니다. 아래는 기존 Task1 사용 설명입니다.

## 기존 Task 1 실행

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
위 Task1 자료와 현재 Task2 실험 공유본은 전체 과제 제출물을 대신하지 않습니다.
