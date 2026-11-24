# Deep Learning 팀 과제 · 진행 공유

실험별 코드·설정·집계 결과와 모델 다운로드를 **`Rkaksdksrud/deep_learning_task`의 `ktg_tried` 브랜치**에서 공유합니다. Task1 자료를 보존하고 Task2 진행 기록을 2026-11-25 결과까지 정리했습니다.

| 구분 | 학습·저장 모델 검증 상태 | 결과 / 모델 |
|---|---|---|
| 기존 Task1 v1 | 기존 기록 보존 | [실험](experiments/task1-v1/README.md) / [모델](models/task1-v1/README.md) |
| Task2 · 11/11 | ResNet18 7개 | [결과](experiments/task2-resnet18-20261111/README.md) / [7개 모델](models/task2-resnet18-20261111/README.md) |
| Task2 · 11/18 | GoogLeNet 1개·VGG11 3개·ResNet18/320 1개 | [결과](experiments/task2-followups-20261118/README.md) / [5개 모델](models/task2-followups-20261118/README.md) |
| Task2 · 11/24 | 무증강 학습 설정 비교 8개 | [결과](experiments/task2-training-settings-20261124/README.md) / [8개 모델](models/task2-training-settings-20261124/README.md) |
| Task2 · 11/25 | 회전 단독·LR 추가 seed·밝기/대비 8개 | [새 결과](experiments/task2-augmentation-lr-20261125/README.md) / [새 모델 manifest](models/task2-augmentation-lr-20261125/manifest.json) |
| Task2 · 초기 원형 | 11/01 ResNet18 baseline, 재검증 완료 | [legacy 모델·provenance](models/task2-legacy-baseline-20261101/README.md) |
| Task2 전체 | **29개 모델 공유 관리** | 기존20개 보존 + 신규8개 + 초기 원형1개 공유 준비 |

새8개와 legacy 원형1개, 총9개 ZIP은 같은 Release 태그 `task2-augmentation-lr-20261125-v1`에 연결합니다. 신규8개 manifest와 [legacy 별도 manifest](models/task2-legacy-baseline-20261101/manifest.json)를 구분합니다. 현재 이 문서는 **모델 공유 준비 상태**를 기록하며 새 Release 게시 성공을 뜻하지 않습니다. 다운로드는 해당 Release가 실제 공개된 뒤 가능합니다. 모든 모델은 단일 환자 holdout 모델이며 **전체 데이터로 재학습한 최종 제출 모델은 아직 없습니다.**

## Task2에서 현재 유지하는 설정

초음파 한 프레임의 Sheath 존재를 `Absent=0 / Present=1`로 분류하며 공식 지표는 이미지 단위 Balanced Accuracy(BA)입니다. 제공 데이터만 사용했고 외부 가중치·추가 라벨·추가 신경망 전처리·앙상블은 사용하지 않았습니다.

현재 참조는 **ResNet18/224 + 학습 전용 수평반전 p=.5 + ±7° 회전 + weighted CE·LR3e-4·WD1e-4·cosine**입니다. random initialization, 고정 .5/.5 정규화와 추론 threshold .5를 유지합니다.

| 같은 fold / seed | 무증강 BA | flip+rotation BA | 차이 |
|---|---:|---:|---:|
| fold0 / 42 | 47.55% | 90.47% | +42.92%p |
| fold0 / 43 | 60.11% | 90.17% | +30.06%p |
| fold1 / 42 | 65.05% | 88.76% | +23.71%p |
| 세 조건 단순 평균 | 57.57% | **89.80%** | **+32.23%p** |

이 평균은 독립3-fold가 아닙니다. 같은 fold0 환자를 두 seed에서 반복했고, checkpoint와 후보 선택에도 사용한 **개발 validation 점수**입니다. 특정 환자의 BA50%·26.25% 실패가 남아 있으며 private/final Test 성능을 뜻하지 않습니다.

## 지금까지 시도한 결과

| 시도 | 실제 결과 | 현재 판단 |
|---|---|---|
| GoogLeNet224 | fold0 BA90.03% | 지정 모델 baseline 확보, 다른 fold 확인 필요 |
| VGG11/224 | LR3e-4에서50.00%; LR1e-4에서80.21%/76.28% | 학습 회복, 두 fold 레시피 평균78.25%; LR matched 개선은 fold0 한 쌍만 확인 |
| ResNet18/320 | fold0 BA89.57%, 224 대비−0.90%p | 추가 fold gate 미달, 224 유지 |
| 무증강 학습 일반 CE | 세 조건 Δ +12.10/+13.23/−5.30%p | 평균+6.68%p여도 seed 악화로 채택 gate 실패 |
| 무증강 LR3e-4→1e-4 | 기존 두 조건 +8.89/+6.24%p에 **11/25 추가 seed +3.83%p** | 세 조건 평균 **+6.32±2.53%p**, 수치 gate 통과; 환자별 큰 악화가 남아 geometry 모델 대체 근거는 아님 |
| 무증강 WD1e-4→0 | fold0 +6.39%p | 단발 탐색, 다른 조건·강도 미확인 |
| 무증강 cosine→constant | fold0 +6.33%p | 단발 탐색, 특정 환자−50%p 악화 |
| 회전 단독: 기존 정책에서 flip 제거 | 세 조건 Δ −0.26/−2.69/−1.37%p | 평균 **−1.44±1.22%p**, 기존 flip+rotation 유지 |
| geometry + 밝기 factor U[.85,1.15] | 세 조건 Δ +1.34/−0.30/−2.10%p | 평균 **−0.35±1.72%p**, 개선 재현 실패 |
| geometry + 대비 factor U[.85,1.15] | fold0 BA91.46%, +0.99%p | 사전 +1%p 추가 확인 gate 미달; 추가 fold/seed 미실행 |

±는 조건 간 차이의 **표본 SD**이며 신뢰구간이 아닙니다. 모델별·설정별 확인 조건 수가 다르고 모든 결과는 개발 validation입니다. 증강을 끈 LR·CE·WD·schedule 비교를 geometry 모델 대비 개선으로 해석하지 않습니다. LR1e-4와 geometry의 결합은 아직 시험하지 않았습니다.

11/25 신규 학습은 **8회·137epoch·56.94분**, 모든 저장 모델을 독립 검증했습니다. LR은 새 seed에서도 상승했지만 해당 조건에서 한 환자 BA가33.75%p 하락했습니다. 수치 gate 통과와 모든 새 환자의 안정성을 구분합니다.

**밝기/대비 증강과 threshold 변경의 공식 허용 여부는 미확정입니다.** 사용자 요청에 따라 탐색했지만 제출 설정 채택에서 제외합니다. 모델의 추론 threshold는 계속 .5입니다. 고정된 기존 geometry3개 모델의 LOPO threshold 진단은 다른 검증환자에서 threshold를 선택해 남은 환자에 적용했으며 Δ +0.90/−0.10/+1.92%p였습니다. checkpoint epoch 자체는 이미 전체 validation에서 선택했으므로 독립 Test나 편향 없는 일반화 평가가 아니고 배포 threshold도 선택하지 않았습니다. 이 진단을 위 fixed-.5 학습 결과와 같은 절차로 순위 매기지 않습니다.

초기 `resnet18_baseline_20261101`은 별도 원형 모델입니다. BA47.5536%, 9epoch/best2이며 기존 독립 재검증을 통과했고, 이번 CPU 비교에서 재학습 대조군과 모든 tensor·저장 예측·split이 일치했습니다. checkpoint bytes는 다르고 당시 source snapshot은 없어 원래 소스를 소급 생성하지 않습니다. 로컬 `runs/` inventory는 full29개+smoke5개이며 smoke는 성과·배포 모델에서 제외합니다(2개 검증 완료, 3개 완료 기록 부족).

[전체 결과 노트북](task2/notebooks/task2_experiment_overview.ipynb), [학습·추론 사용법](task2/README.md), [11/25 버전 코드](task2/experiments/20261125/README.md), [자료 검토와 다음 시도](experiments/task2-resnet18-20261111/RESEARCH_AND_NEXT_STEPS.md)를 참고하세요. 문헌은 기존 검토 기록을 인용하며 이번 공유 작업에서 새 검색이나 논문 성능 재현을 수행하지 않았습니다.

## 팀원이 코드와 모델 받기

기존 작업 폴더와 분리해 받는 예시입니다.

```bash
git clone --branch ktg_tried --single-branch https://github.com/Rkaksdksrud/deep_learning_task.git deep_learning_task-ktg
cd deep_learning_task-ktg
python models/task2-resnet18-20261111/download.py --list
python models/task2-followups-20261118/download.py --list
python models/task2-training-settings-20261124/download.py --list
python models/task2-augmentation-lr-20261125/download.py --list
python models/task2-legacy-baseline-20261101/download.py --list
```

모든 실제 try를 결과가 나빠도 별도 ZIP으로 보관하며, 미실행 template에는 가중치나 점수가 없습니다. 각 manifest에 ZIP·모델 SHA256·지표를 연결하고 가중치 ZIP은 Git 대신 Releases에 둡니다. 여러 모델의 공유는 앙상블 사용을 뜻하지 않습니다.

```bash
# 새 Release 공개 후, 11/25 추가 LR seed 모델 하나를 받는 예시
python models/task2-augmentation-lr-20261125/download.py --run resnet18_20261125_lr1e4_f0_s43 --output downloads/task2/lr1e4-s43
python -m pip install -r downloads/task2/lr1e4-s43/requirements.txt
python downloads/task2/lr1e4-s43/predict.py --images input_images --output predictions.csv --device cpu
```

새8개 checkpoint의 공개 복사본은 개인 경로가 든 `data_root`/`output_root` 두 값만 팀 경로로 바꿉니다. 원본 로컬 checkpoint는 보존하고 **원본 SHA와 공개 SHA를 구분**합니다. tensor·추론 동등성과 패키지 검증 범위는 새 manifest/검증 기록을 따릅니다. 기존20개 공개 모델은 변경하지 않습니다. best checkpoint는 추론용이며 optimizer/scheduler/RNG 재개 state는 공유하지 않습니다.

각자 허가받은 과제 데이터는 `data/task2/`에 따로 준비합니다. 원본 영상·라벨·환자/표본 목록·표본별 예측·분할표·과제 PDF·영상 포함 노트북은 공유하지 않습니다. 공개 노트북은 집계 지표와 코드 안내만 포함합니다. 재학습은 새 run 이름으로 수행하고 검증 결과·코드 버전을 별도 기록합니다.

현재 Task1 코드/모델/결과는 보존했습니다. 아래는 기존 Task1 사용 설명입니다.

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
