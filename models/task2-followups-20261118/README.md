# Task2 후속 실험 모델 — 2026-11-18

GoogLeNet224, VGG11·224와 ResNet18·320의 **전체5개 run을 학습하고 저장 checkpoint를 독립 재검증**했습니다. 성공·실패한 시도를 모두 개별 ZIP으로 남깁니다. 실제 포함 모델·파일 크기·SHA256은 [manifest.json](manifest.json)에 기록했습니다. 링크는 해당 Release가 게시된 이후 동작합니다.

- [후속 Release](https://github.com/Rkaksdksrud/deep_learning_task/releases/tag/task2-followups-20261118-v1)
- [전체 비교와 채택·기각 이유](../../experiments/task2-followups-20261118/README.md)
- 이전 ResNet18 7개 모델: [기존 배포본](../task2-resnet18-20261111/). 기존 파일·manifest·Release는 보존합니다.
- 각 모델은 서로 독립적으로 학습한 단일 CNN입니다. 모델들을 결합해 예측하지 않습니다.
- 환자별 Train/Valid 분할의 holdout 모델입니다. 전체 제공 데이터 최종 제출 모델이나 private Test 점수가 아닙니다.

## 다운로드와 추론

저장소 루트에서:

```bash
python models/task2-followups-20261118/download.py --list
python models/task2-followups-20261118/download.py \
  --run googlenet_20261118_geometry_f0_s42 --output downloads/googlenet-followup
python -m pip install -r downloads/googlenet-followup/requirements.txt
python downloads/googlenet-followup/predict.py --images input_images --output predictions.csv --device cpu
```

CUDA가 준비된 환경에서는 `--device cuda`를 사용할 수 있습니다. ZIP을 이미 받았다면 downloader의 `--archive ZIP_PATH`로 네트워크 없이 검증·압축해제를 할 수 있습니다. 기존 출력 디렉터리와 예측 CSV는 덮어쓰지 않습니다.

다운로더는 ZIP 크기·SHA256, 정확한 여섯 개 파일, 내부 체크섬 목록의 전체 커버리지와 모든 파일 해시, 모델 파일명과 원본 checkpoint 해시를 확인합니다. 경로 탈출·중복 ZIP entry·symlink·예상치 못한 추가 파일은 거부합니다.

각 ZIP의 파일은 다음 여섯 개입니다.

| 파일 | 내용 |
|---|---|
| `model.pt` | 원본 추론용 checkpoint 바이트 그대로. 모델 계열은 내부 metadata로 구분 |
| `predict.py` | ResNet18/VGG11/GoogLeNet 중 checkpoint에 맞는 모델 하나를 로드하는 독립 추론 코드 |
| `requirements.txt` | 고정한 Python 패키지 버전 |
| `metadata.json` | 집계 검증지표, 학습설정·시간, 선택epoch, 코드·모델 해시 |
| `README.md` | 해당 모델 사용법과 해석 범위 |
| `SHA256SUMS.json` | 나머지 다섯 파일의 SHA256 |

추론은 RGB/bilinear letterbox, 고정 normalization0.5/0.5, 학습된 입력 해상도, 임계값0.5를 사용합니다. 학습 augmentation은 적용하지 않습니다. 출력 컬럼은 `sample_id,path,probability,prediction`이며 Present=1, Absent=0입니다. GoogLeNet은 eval 상태에서 주 출력만 사용합니다. 다른 backend에서의 원래 GPU 확률 bit parity를 보장하지 않습니다.

## 포함 기준과 제외 항목

학습 summary와 독립 checkpoint 재검증의 `all_checks_passed=true`, 배포할 checkpoint SHA256에 연결된 재검증 기록을 확인한 다음 모델만 포함했습니다. 미실행 run은 manifest의 `omitted_runs`에 명시했습니다.

| run ID / ZIP | Valid BA | 해석 |
|---|---:|---|
| [googlenet_20261118_geometry_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-followups-20261118-v1/googlenet_20261118_geometry_f0_s42.zip) |90.0262%|첫 full baseline. 추가 fold·seed 검증 필요|
| [vgg11_20261118_geometry_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-followups-20261118-v1/vgg11_20261118_geometry_f0_s42.zip) |50.0000%|lr3e-4 학습 실패, 전부 Absent. 실패 진단용|
| [resnet18_20261118_320_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-followups-20261118-v1/resnet18_20261118_320_f0_s42.zip) |89.5709%|입력 크기만 변경. 기존224보다0.8982%p 하락해 기각|
| [vgg11_20261118_lr1e4_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-followups-20261118-v1/vgg11_20261118_lr1e4_f0_s42.zip) |80.2102%|lr1e-4로 학습 회복, 과적합·환자별 실패 남음|
| [vgg11_20261118_lr1e4_f1_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-followups-20261118-v1/vgg11_20261118_lr1e4_f1_s42.zip) |76.2810%|같은 레시피의 추가 fold. 고학습률 fold1 대조군은 없음|

ResNet320의 fold1은 사전에 정한 fold0 개선 기준을 충족하지 못해 **실행하지 않았습니다**. 남은 비교 예산으로 VGG lr1e-4의 추가 환자 fold를 확인했습니다. VGG 두 fold 평균은78.2456%이며 독립 Test 성능이나 학습률 효과의 두 fold 평균이 아닙니다. 현재 참조 설정은 [기존 ResNet18·224](../task2-resnet18-20261111/README.md)입니다.

[패키지 검증 기록](package_validation.json)은 모든 ZIP의 파일 해시·원본 checkpoint 일치·합성영상의 원본/독립 추론 parity를 확인한 결과입니다. 실제 holdout 재검증은 [배포 가중치 SHA에 연결된 기록](../../experiments/task2-followups-20261118/release_validation.json)에서 별도로 확인합니다. 합성영상 점검을 데이터 성능 평가로 해석하지 않습니다.

데이터·이미지·과제 PDF·원래 labels CSV·환자 ID·분할표·표본별 예측·이미지가 내장된 노트북은 포함하지 않습니다. optimizer/scheduler/RNG가 있는 `last_training_state.pt`도 포함하지 않습니다. 새 best 모델과 이전7개 모델 모두 추론용 checkpoint이므로 그것만으로 학습 재개를 할 수 없습니다. 신규 코드의 로컬 재개 기능은 [Task2 README](../../task2/README.md)를 참고하세요.

빌드 명령은 저장소 루트에서 다음과 같습니다. 출력 디렉터리는 기존 파일을 보존하기 위해 새 경로여야 하며 Git 저장소 밖에 둡니다.

```bash
python tools/build_task2_followup_packages.py \
  --source-task2 '/path/to/original/task2' \
  --output '/path/outside/repository/task2-followups-20261118-v1'
```

이 명령은 로컬 ZIP과 manifest를 생성합니다. Git push나 GitHub 업로드는 수행하지 않습니다.
