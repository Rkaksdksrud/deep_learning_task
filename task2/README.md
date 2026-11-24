# Task2: 지정 CNN 실험 코드와 사용 방법

초음파 한 프레임의 Sheath를 `Absent=0 / Present=1`로 분류한다. 공식 지표는 이미지 기준 Balanced Accuracy(BA)다. 제공 데이터만 사용하며 random initialization, 단일 모델, 임계값0.5를 유지했다. 외부 가중치·다른 신경망 전처리·앙상블·TTA는 사용하지 않았다.

11월11일 **ResNet18 7개**에 이어 11월18일 **GoogLeNet 1개·VGG11 3개·ResNet18·320 1개**의 full 학습과 저장가중치 검증을 완료했다. 모두 환자 holdout 모델이며 전체2,948장으로 재학습한 최종 제출 모델이 아니다. [후속 실험 결과](../experiments/task2-followups-20261118/README.md)와 [후속5개 모델](../models/task2-followups-20261118/README.md)을 함께 확인한다.

11월24일에는 **무증강 ResNet18 학습 설정 8개**를 추가 실행·독립 검증했다. [설정 비교](../experiments/task2-training-settings-20261124/README.md)와 [8개 모델](../models/task2-training-settings-20261124/README.md)은 이전 증강 실험과 별개다. 일반 CE는 추가 seed에서 악화해 채택 gate를 통과하지 못했다.

11월25일에는 **회전 단독3개·무증강 LR 추가 seed1개·밝기3개·대비1개**, 총8개 full run/137epoch/56.94분을 추가했고 모두 독립 검증했다. 기존20개와 신규8개 외에 초기 원형1개를 별도로 보존해 총29개 모델을 공유 관리한다. [최신 실험](../experiments/task2-augmentation-lr-20261125/README.md) · [새8개 모델 manifest](../models/task2-augmentation-lr-20261125/manifest.json) · [집계 노트북](notebooks/task2_experiment_overview.ipynb).

- 회전 단독은 기존 flip+rotation보다 세 조건 모두 낮아 평균−1.44±1.22%p다. flip을 제거한 정책 비교이며 회전의 독립 기여나 flip-only 비교가 아니다.
- 무증강 LR1e-4는 새 seed에서도+3.83%p, 세 조건 평균+6.32±2.53%p로 수치 gate를 통과했다. 그러나 한 환자 BA가33.75%p 악화했고 geometry에 낮춘 LR을 적용한 조합은 시험하지 않았다. **기존 geometry/LR3e-4 참조 모델은 유지한다.**
- 밝기는 세 조건 평균−0.35±1.72%p로 개선이 유지되지 않았고, 대비는 단발+0.99%p로 추가 확인 gate 미달이다. 밝기/대비와 threshold 변경은 공식 허용 미확정인 탐색 기록이며 제출 설정 채택에서 제외한다. 새 모델 추론 threshold는.5다.
- 기존 geometry 모델의 LOPO threshold 진단은 checkpoint 선택에 사용한 validation을 다시 쓰는 조건부 분석이다. 독립 Test·nested 검증·환자별 배포 정책이 아니며 단일 배포 threshold를 고르지 않았다.

새8개 모델은 **공유 준비 상태**다. Release 태그 `task2-augmentation-lr-20261125-v1`의 실제 게시 확인은 모델 manifest/배포 기록을 따른다. 새 공개 checkpoint는 개인 경로 두 값만 portable하게 바꾸므로 원본/공개 SHA가 다를 수 있고, 원본 로컬 모델과 tensor·추론 동등성 검증을 구분한다. 기존20개 모델은 보존한다.

## 초기 baseline 원형과 전체 inventory

[11/01 legacy baseline](../models/task2-legacy-baseline-20261101/README.md)은 새 학습8개에 포함하지 않는 초기 full run이다. BA47.5536%, 9epoch/best2, 학습216.043초이며 기존 독립 재검증이 통과했다. 이번 CPU 비교에서는11/11 재학습 대조군과 모든 tensor·저장 예측·split bytes가 일치했다. 원본 checkpoint bytes/SHA는 다르고 당시 source snapshot이 없어 원래 실행 소스라고 소급 주장하지 않는다.

이번 Release 태그 `task2-augmentation-lr-20261125-v1`에 신규8개와 legacy1개 총9개 ZIP을 연결하고, [legacy 별도 manifest](../models/task2-legacy-baseline-20261101/manifest.json)를 제공한다. 공유 준비 상태이며 실제 게시·패키지 검사 범위는 해당 모델 기록을 따른다. 로컬 `runs/`는 full29개+smoke5개다. smoke는 성과·배포 모델에서 제외하며2개는 기능 검증 완료,3개는 완료 기록 부족이다. [분류 근거](../experiments/task2-augmentation-lr-20261125/README.md)를 참고한다.

## 환경과 데이터 위치

원래 학습 환경: Python3.10.13, torch2.10.0+cu128, torchvision0.25.0+cu128, NumPy1.26.4, pandas2.3.3, Pillow12.1.0, scikit-learn1.7.2, RTX4060 Laptop GPU. 아래 명령은 **저장소 루트**에서 실행한다. Task1과 별도 가상환경을 권한다. `task2/requirements.txt`는 기존 Task1의 루트 requirements를 변경하지 않는다.

```bash
python -m venv .venv-task2
# Linux/macOS: source .venv-task2/bin/activate
# Windows PowerShell: .venv-task2\Scripts\Activate.ps1
python -m pip install -r task2/requirements.txt
```

Windows에서 시간대 데이터가 없으면 `python -m pip install tzdata`가 필요하다. CUDA용 PyTorch 설치 방식은 사용자의 운영체제/GPU에 맞추고 실제 버전을 기록한다. CPU 추론도 지원한다. 이 코드는 기존 Task1 스크립트처럼 자동으로 다른 Python 환경으로 전환하지 않는다.

원본 영상·labels.csv는 저장소나 Release에 포함하지 않는다. 수업에서 허가받은 데이터를 각자 다음 위치에 준비한다.

```text
data/task2/
├── labels.csv              # sample_id,path,label,patient_group
└── images/                 # CSV path가 images/<id>.png인 원본 영상
```

기존 `task2/code`의 `data_root`와 `output_root` 상대 경로는 현재 터미널 위치가 아니라 `task2/` 기준이다. 11/25 버전 소스는 아래 별도 기준 경로를 사용한다. 세 config의 기본값은 `../data/task2`, `../runs/task2`다. 학습 결과는 저장소의 `runs/task2/<run-name>/` 아래에 생긴다.

## 11월11일 ResNet18 7개 체크포인트와 결과

다운로드 링크·Release asset 목록·해시·실험 원본 기록은 저장소의 Task2 공유 안내를 따른다. 모델 묶음을 받은 뒤 아래와 같이 각 원래 run 폴더를 `runs/task2/` 아래에 둔다. 모든 모델을 평균하거나 투표하지 않고 필요한 **checkpoint 하나**를 고른다.

| run 이름 (`resnet18_20261111_` 뒤) | 변경 | fold/seed | 선택 epoch | Valid BA |
|---|---|---|---:|---:|
| control_f0_s42 | 무증강·가중 CE |0/42|2|47.5536%|
| plain_ce_f0_s42 | 무증강·일반 CE |0/42|20|59.6532%|
| geometry_f0_s42 | 기하 정책·가중 CE |0/42|6|90.4691%|
| control_f0_s43 | 무증강·가중 CE |0/43|2|60.1104%|
| geometry_f0_s43 | 기하 정책·가중 CE |0/43|8|90.1697%|
| control_f1_s42 | 무증강·가중 CE |1/42|1|65.0465%|
| geometry_f1_s42 | 기하 정책·가중 CE |1/42|7|88.7566%|

기하 정책과 무증강을 맞춘 세 조건 평균 BA는57.5702%→89.7985%(+32.2283%p)다. fold0을 두 seed로 반복했으므로 독립3-fold가 아니다. 방법·checkpoint 선택에 사용한 validation 점수이며 private Test 점수가 아니다. fold0의 한 환자는 기하 정책 두 seed에서 전부 Present(환자 BA50%), fold1의 한 환자는 BA26.25%로 실패가 남는다.

현재 참조 모델은 사전 중심 조건인 `resnet18_20261111_geometry_f0_s42/resnet.pt` 하나다. 다른 checkpoint는 비교·진단용으로 함께 제공한다.

## 재학습 없이 폴더 예측

```bash
python task2/code/predict.py \
  --weights runs/task2/resnet18_20261111_geometry_f0_s42/resnet.pt \
  --images /absolute/path/to/your/images \
  --output runs/task2/predictions_geometry_f0_s42.csv \
  --device cpu
```

PowerShell에서는 줄 끝 `\`를 쓰지 말고 한 줄로 실행하거나 PowerShell의 줄 이어쓰기 문법을 사용한다. CUDA가 준비됐다면 `--device cuda`를 사용한다. `sample_id,path,probability,prediction`을 저장하며 `prediction`이0/1 출력이다. 입력 폴더에 라벨 CSV는 필요 없다. 추론은 checkpoint의 image_size와 threshold를 사용하고 학습 증강을 적용하지 않는다. 기존 출력 파일은 덮어쓰지 않는다.

## Release만 받은 경우의 환자 holdout 평가

Release ZIP에는 `resnet.pt`, 독립 예측 코드, 환경·요약 metadata가 있다. **원본 데이터, split_manifest.csv, valid_predictions.csv 같은 표본별 기록은 공개 배포하지 않는다.** 예측에는 `.pt`와 이미지 폴더만 있으면 되지만, holdout 평가에는 수업에서 허가받은 원본 데이터 전체를 팀원이 별도로 보유해야 한다.

```bash
python task2/code/evaluate_released.py \
  --weights runs/task2/resnet18_20261111_geometry_f0_s42/resnet.pt \
  --data-root ../data/task2 \
  --output-dir runs/task2/evaluation_geometry_f0_s42 \
  --device cpu
```

이 명령은 원본 labels.csv의 SHA256과 전체 영상의 byte-level collection hash를 확인하고 checkpoint의 n_splits/val_fold로 환자 split을 재생성한다. 고정 checkpoint를 eval 모드로 적용해 checkpoint 안에 저장된 **전체 BA·혼동행렬·표본 수**와 비교한다. 원래 저장 확률·표본별 예측 parity는 확인할 수 없다. 해당 항목은 결과에 `False`로 명시한다. 원래 sklearn 버전 등 `task2/requirements.txt` 환경을 사용해야 같은 split을 기대할 수 있다. CSV의 줄바꿈을 다시 저장해 byte가 달라진 경우에도 SHA 검사는 거절하므로 원본 파일을 사용한다.

`--data-root ../data/task2`는 `task2/` 기준으로 해석되어 저장소의 `data/task2`를 가리킨다. 절대 경로도 가능하다. `--weights`·`--output-dir`·`predict.py --images`는 터미널의 현재 위치 기준이다. 기존 출력 폴더는 덮어쓰지 않는다. 이 evaluator는 학습·임계값 선택·분할 탐색을 하지 않는다.

## 원본 run 기록을 별도로 보유한 경우의 strict 재검증

`validate.py`는 가중치 옆의 원래 `split_manifest.csv`, `valid_predictions.csv`를 대조하는 도구다. 이 두 비공개 파일은 Release에 없으므로 **Release만 받은 상태로 이 명령을 실행할 수 없다.** 원본 run 파일을 로컬에 이미 보유한 팀원의 감사 경로다. 비공개 파일을 GitHub에 추가하지 않는다.

배포한 checkpoint의 `config.data_root`는 원래 프로젝트 값인 `data/raw`다. checkpoint를 수정하거나 새 기본 경로로 자동 대체하지 않는다. 원본 기록을 갖고 있다면 다음과 같이 실제 데이터 위치를 명시한다.

```bash
python task2/code/validate.py \
  --weights runs/task2/resnet18_20261111_geometry_f0_s42/resnet.pt \
  --data-root ../data/task2 \
  --output-dir runs/task2/strict_revalidation_geometry_f0_s42 \
  --device cuda
```

validation 출력에는 원래 checkpoint의 경로 설정과 실제 사용한 경로·override 여부가 기록된다. CSV의 ID/path/label/patient가 원래 manifest와 다르면 중단하며 분할을 다시 고르지 않는다.

GPU 저장 확률과의 strict 비교는 원본처럼 `atol=rtol=1e-5`다. CPU/다른 backend는 이진 예측·BA가 같아도 확률 차이 때문에 이 strict 검사에 실패할 수 있다. 기존 무증강 모델의 CPU/GPU 최대 확률 차이는0.002289였다. strict 검증의 확률 parity와 `evaluate_released.py`의 aggregate 비교를 구분한다. 실패를 자동 성공 처리하거나 허용오차를 몰래 완화하지 않는다.

## 학습 코드와 설정

새 학습이 필요할 때만 아래 명령을 실행한다. 공유한 가중치를 다운로드해서 예측하는 데 학습은 필요 없다.

```bash
python task2/code/resnet.py --config task2/configs/baseline.json --device cuda --run-name resnet_new_baseline
python task2/code/resnet.py --config task2/configs/geometry.json --device cuda --run-name resnet_new_geometry
python task2/code/resnet.py --config task2/configs/plain_ce.json --device cuda --run-name resnet_new_plain_ce
```

| config | class_weighted_loss | geometric_augmentation |
|---|---|---|
| baseline.json |true|false|
| geometry.json |true|true|
| plain_ce.json |false|false|

기본은 fold0/seed42다. 다른 조건은 JSON의 `val_fold`/`seed`를 명시적으로 바꾸고 새 run 이름을 사용한다. split은 sample_id 정렬 후 `StratifiedGroupKFold(5,shuffle=False)`로 고정하며 환자 중복을 검사한다. 정규화는 고정0.5/0.5이며 클래스 가중치는 Train 클래스 수로만 계산한다.

기하 정책은 **학습에만** thumbnail→수평반전p.5→uniform[-7,7]° 회전(expand=True)→224 letterbox다. 회전 후 축소·추가 보간도 포함하므로 개별 flip/회전의 효과를 분리한 결과가 아니다. validation/inference는 RGB·bilinear letterbox·[-1,1] 고정 전처리다.

기본 학습은 batch16, AdamW lr3e-4/WD1e-4, FP32, cosine 최대30epoch, patience7다. Valid BA 최대, 동률이면 Valid loss 최소로 checkpoint 하나를 고른다. 소규모 경로 점검은 `--smoke-test --device cpu`이며 해당 결과는 성능 보고·제출용으로 쓰지 않는다.

VGG11·GoogLeNet은 각각 `task2/code/vgg.py`, `task2/code/googlenet.py`로 독립 학습한다. GoogLeNet 학습은 두 보조 손실을 각각0.3배 더하며 추론은 주 출력만 쓴다. 11월18일 GoogLeNet224 fold0 BA90.03%를 얻었다. VGG11은 기본lr3e-4에서 BA50%로 학습에 실패했으며, **lr1e-4**에서 fold0/1 BA80.21%/76.28%로 회복했다. ResNet320은224를 개선하지 못했다. ResNet의 증강 개선량이 다른 구조에도 그대로 전이된다고 가정하지 않는다.

실제 후속 run의 기록과 경로만 이 저장소에 맞춘 재실행 config는 `experiments/task2-followups-20261118/run_records/<run ID>/`에 있다. VGG의 작동한 설정을 다시 학습하려면 새 run 이름으로 다음을 사용한다.

```bash
python task2/code/vgg.py --config experiments/task2-followups-20261118/run_records/vgg11_20261118_lr1e4_f0_s42/config.json --device cuda --run-name vgg_new_lr1e4_f0
```

이 명령의 결과가 기존 점수와 정확히 같다는 보장은 없다. CUDA 비결정성과 제한된 환자 수·검증 기반 선택의 영향을 함께 고려한다.

## 2026-11-25 별도 버전 코드와 재실행

기존 `task2/code`와 이전 설정을 보존하고 이번 실험은 [task2/experiments/20261125](experiments/20261125/README.md)에 분리했다. `code/`, `configs/`, `analysis/`에 해당 버전의 학습·추론 소스, 재실행 설정, cached CSV용 threshold 분석을 둔다. 실제 학습에서 고정한 원래13개 source의 hash 목록을 보존하고, 환자별 로컬 감사용3개 스크립트는 공개하지 않는다. 공개용 일반 소스10개와 팀 경로 config를 원래 학습 snapshot/config와 구분한다.

| 새 옵션 | 기본값 | 이번 비교 |
|---|---|---|
| `geometric_horizontal_flip` | `true` | `false`는 flip 적용만 끄고 기존 flip decision RNG draw와 회전 각도 경로를 유지 |
| `brightness_jitter` | `0.0` | `.15`는 Train에서 실영상 영역 brightness factor U[.85,1.15] 적용 |
| `contrast_jitter` | `0.0` | `.15`는 Train에서 실영상 영역 contrast factor U[.85,1.15] 적용 |

광도 변환은 회전/검정 padding 전에 적용하며 brightness와 contrast를 결합하지 않았다. Valid/추론에는 어느 증강도 적용하지 않는다. 기본값0은 기존 변환/RNG 경로를 보존한다. 합성 CPU14검사와4개 모드 GPU smoke를 통과했지만 이를 과제 성능으로 보고하지 않는다.

**새 버전 config의 경로 기준은 `task2/experiments/20261125/`**다. `data_root=../../../data/task2`, `output_root=../../../runs/task2/20261125`이며 모두 저장소의 데이터/새 run 디렉터리로 연결된다. 저장소 루트에서:

```bash
python task2/experiments/20261125/code/resnet.py --config task2/experiments/20261125/configs/rotation_f0_s42.json --device cuda --run-name your_unique_rotation_run
```

LR/brightness/contrast 또는 다른 fold/seed는 해당 버전 config를 선택하고 새 run 이름을 쓴다. config10개 중 대비 추가 fold/seed2개는 사전 gate로 **실행하지 않은 template**이며 기존 결과나 가중치가 없다. 이 예시를 전부 재실행하거나 최종 제출용 설정으로 자동 채택하라는 뜻은 아니다.

원본 영상·환자별 cached prediction은 공유하지 않는다. `analysis/`의 threshold 도구를 각자 허가받은 로컬 prediction CSV에 적용할 때도 다른 환자에서 선택한 전역 threshold의 LOPO 진단으로만 해석한다. 기존 checkpoint의 validation epoch 선택 의존성은 남고, 공개 모델의 기본 threshold.5를 바꾸지 않는다.

새 모델 다운로드·추론은 [모델 manifest](../models/task2-augmentation-lr-20261125/manifest.json)를 따른다. 새 checkpoint의 공개 복사본은 데이터/출력 경로만 바꾼 것으로, 모델의 원본 SHA와 배포 SHA를 따로 기록한다. 로컬 원본 재개 state와 공개 추론 checkpoint는 다르며 새 버전 소스로 과거 state를 강제 재개하지 않는다.

## 2026-11-18 추가: epoch 경계 학습 재개

새 학습 코드는 매 epoch 완료 시 별도 `last_training_state.pt`에 model/optimizer/scheduler, 완료epoch/history, 최적epoch·지표, patience counter, Python/NumPy/torch CPU/CUDA RNG와 두 DataLoader generator를 원자 저장한다. 기존 추론용 `resnet.pt`/`vgg.pt`/`googlenet.pt` 형식과 학습·선택 정책은 유지한다.

```bash
python task2/code/googlenet.py --config task2/configs/geometry.json --device cuda --run-name googlenet_new_geometry
python task2/code/googlenet.py --config task2/configs/geometry.json --device cuda --resume runs/task2/googlenet_new_geometry
```

`--resume`는 run 디렉터리 또는 그 안의 `last_training_state.pt`를 받는다. `--run-name`과 함께 쓰지 않는다. 최초 실행의 config와 CLI override를 동일하게 지정해야 하며, config/model/smoke/split/source snapshot/환경이 다르면 중단한다. 원래 run의 source를 덮어쓰지 않고 명시한 run만 이어 쓴다. epoch 도중 중단되면 마지막 완료 epoch 다음부터 다시 실행한다. max epoch/patience에 이미 도달했다면 추가 학습하지 않는다.

**기존7개 모델과 새로 배포하는 best 모델은 모두 추론용 checkpoint다.** optimizer/scheduler/RNG를 담는 `last_training_state.pt`는 로컬 학습 재개 전용으로 Git/Release에 올리지 않는다. 추론용 best 가중치만으로 정확한 재개를 시도하면 거부한다.

작은 Conv+BN+Dropout 모델과 합성 RGB 영상으로 CPU 검증을 수행했다. 변경 전 코드와 새 코드의 연속 학습, 연속4epoch와 epoch2 원자 커밋 직후 중단→재개에서 선택 가중치·수학적 지표·optimizer/scheduler·RNG가 정확히 일치했다. 시간·처리량은 비교에서 제외했다. 이는 과제 모델의 성능이나 CUDA 재개 수치 일치 검증이 아니다. GoogLeNet/VGG11의 adaptive-pooling CUDA backward에서 비결정적 경고를 관측했으므로 기존 `warn_only=True`를 유지하며 GPU bitwise 일치를 주장하지 않는다.

## 원본 코드·config와의 차이 및 한계

11월24일 학습 설정 비교를 위해 다음 옵션을 추가했다. [무증강 비교 기록](../experiments/task2-training-settings-20261124/README.md)과 [해당 모델 묶음](../models/task2-training-settings-20261124/README.md)은 이전 증강 실험과 별도 버전이다. 기존 모델·실험 기록은 보존한다.

| 옵션 | 기본 동작 | 비교에서의 사용 |
|---|---|---|
| `lr_scheduler` | 미지정 시 기존 `cosine` | `constant`는 초기 학습률을 매 epoch 유지 |
| `validation_class_weighted_loss` | 미지정 시 학습 가중치 선택을 따름 | `true`로 고정하여 모든 후보의 검증 loss와 BA 동률 판정 기준을 동일하게 유지 |

새 비교의 `class_weighted_loss=false`는 **학습 손실만 일반 CE**로 바꾼다. checkpoint의 `class_weights`는 계속 학습용이고, 명시된 `validation_class_weights`는 Train 클래스 수에서 계산한 평가용이다. 독립 검증기는 새 평가용 값이 있으면 사용하고, 과거 checkpoint는 기존 값으로 처리한다. 정규화·임계값·모델 구조와 기본 cosine 동작은 유지했다. CPU 합성17검사 및 GPU 소규모 학습/독립 재로딩 검증을 완료했다. 테스트 점수를 과제 성능으로 보고하지 않는다.

기존 epoch 경계 state를 재개할 때는 당시 source/config/environment가 일치해야 한다. 새 소스로 과거 state를 강제로 이어 학습하지 않는다. 저장된 추론 checkpoint를 불러 예측하는 경로는 유지한다.

- `common.py`, `data.py`, 세 모델 진입점, `predict.py`는 원본 알고리즘을 유지한다. 배포 config는 경로만 이 저장소 구조에 맞췄고 policy/loss 플래그를 명시했다.
- `validate.py`에는 명시적인 `--data-root` override와 사용 경로 기록을 추가했다. 학습·metric·모델 선택·전처리 계산은 변경하지 않았다. 새 배포 소스를 과거 checkpoint의 실제 실행 snapshot과 동일하다고 주장하지 않는다.
- 기존7개 checkpoint의 원래 config·환경·epoch·preprocessing metadata는 보존한다. 배포 config는 재실행 템플릿이며 과거 run의 원본 config를 대체하지 않는다. fold/seed·class_weighted_loss·geometry 여부는 각 run 기록을 확인한다.
- 기존7개 중6개와 후속5개 실험은 당시 source snapshot/해시가 있고, 최초 대조군 C0는 snapshot 보강 전 실행으로 별도 보관 코드·재현 parity 기록이 있다. 확인 가능한 범위를 넘어 소스 provenance를 소급 생성하지 않는다.
- 배포한 best checkpoint는 모델 state_dict(가중치·BN buffer)와 선택epoch/config만 담고 optimizer/scheduler/RNG는 없다. 신규 학습의 별도 로컬 `last_training_state.pt`가 있을 때만 위 epoch 경계 재개를 지원한다. 선택epoch는 재개 가능한 마지막epoch를 뜻하지 않는다.
- 제공 데이터 외 학습, 추가 신경망, 사전학습, 앙상블은 사용하지 않는다. 11/25 별도 버전에는 공식 허용 미확정인 밝기/대비 및 threshold 조건부 진단을 탐색 기록으로 공유하지만 제출 설정 채택에서 제외한다. CLAHE·새ROI·구조 변경·BN 재보정 등은 적용하지 않았다.
