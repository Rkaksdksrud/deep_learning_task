# Task2 학습 설정 비교 모델 — 2026-11-24

**무증강 ResNet18의 새 full run 8개를 모두 학습하고 저장 checkpoint를 독립 검증했습니다.** 모든 실제 시도는 결과가 나빠도 별도 ZIP으로 보존합니다. 5개 조건부 template는 사전 예산에 따라 미실행했으며 점수를 부여하지 않습니다. 파일 크기·원본 가중치 및 ZIP SHA256은 [manifest.json](manifest.json)에 있습니다.

현재 공유 상태: 로컬 8개 ZIP 빌드와 실제 모델 CPU 패키지 검증 완료, Release 게시 확인 중입니다. 아래 URL은 Release 공개 후 사용할 수 있습니다.

- [학습 설정 비교 Release](https://github.com/Rkaksdksrud/deep_learning_task/releases/tag/task2-training-settings-20261124-v1)
- [결과·설정·채택 판단](../../experiments/task2-training-settings-20261124/README.md)
- 이전 [ResNet18 7개](../task2-resnet18-20261111/README.md), [후속 5개](../task2-followups-20261118/README.md)는 보존합니다.

무작위 초기화·기본 ResNet18·224 입력·임계값0.5를 고정하고, **수평반전과 회전을 모두 껐습니다.** reference는 weighted CE·lr3e-4·weight decay1e-4·cosine입니다. 각 실험은 네 설정 중 하나만 바꾸며 검증 loss와 동점 checkpoint 선택은 Train 수로 구한 동일 weighted CE입니다.

일반 CE는 두 fold에서 +12.10/+13.23%p였지만 추가 seed에서 -5.30%p여서 사전 채택 gate를 통과하지 못했습니다. lr1e-4는 두 fold 평균 +7.56%p로 추가 확인 우선 후보이나 seed 검증이 없고 fold0 양성 recall19.76%가 약점입니다. WD0과 constant는 단일 조건 결과입니다. 조합·최적 강도·증강 재도입 효과를 검증하지 않았습니다. 기존 기하 증강 ResNet18 참조 모델 선택은 유지합니다.

| run ID / ZIP | Valid BA | 양성 / 음성 recall |
|---|---:|---:|
| [resnet18_20261124_reference_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_reference_f0_s42.zip) | 47.55% | 88.02% / 7.08% |
| [resnet18_20261124_unweighted_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_unweighted_f0_s42.zip) | 59.65% | 63.47% / 55.83% |
| [resnet18_20261124_lr1e4_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_lr1e4_f0_s42.zip) | 56.44% | 19.76% / 93.12% |
| [resnet18_20261124_wd0_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_wd0_f0_s42.zip) | 53.94% | 24.55% / 83.33% |
| [resnet18_20261124_constant_f0_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_constant_f0_s42.zip) | 53.89% | 55.69% / 52.08% |
| [resnet18_20261124_unweighted_f1_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_unweighted_f1_s42.zip) | 78.27% | 79.26% / 77.29% |
| [resnet18_20261124_unweighted_f0_s43](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_unweighted_f0_s43.zip) | 54.81% | 61.08% / 48.54% |
| [resnet18_20261124_lr1e4_f1_s42](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-training-settings-20261124-v1/resnet18_20261124_lr1e4_f1_s42.zip) | 71.28% | 53.19% / 89.38% |

모두 환자 분리 개발용 validation의 단일 holdout 모델입니다. private Test 성능이나 전체 데이터 최종 제출 모델이 아닙니다. 여러 모델을 합쳐 예측하지 않습니다.

## 다운로드와 추론

저장소 루트에서 실행합니다.

```bash
python models/task2-training-settings-20261124/download.py --list
python models/task2-training-settings-20261124/download.py \
  --run resnet18_20261124_lr1e4_f1_s42 --output downloads/task2-settings/lr1e4-f1
python -m pip install -r downloads/task2-settings/lr1e4-f1/requirements.txt
python downloads/task2-settings/lr1e4-f1/predict.py \
  --images input_images --output predictions.csv --device cpu
```

CUDA가 있으면 `--device cuda`를 사용할 수 있습니다. 내려받은 ZIP은 `--archive ZIP_PATH`로 오프라인 검증할 수 있습니다. 기존 출력 디렉터리·예측 CSV는 덮어쓰지 않습니다. 확률은 다른 backend에서 원래 GPU와 bit 단위로 같음을 보장하지 않습니다.

다운로더는 ZIP 크기·SHA256, 정확히 여섯 개의 flat 파일, 모든 내부 체크섬과 원본 checkpoint 해시를 확인합니다.

| 파일 | 내용 |
|---|---|
| `model.pt` | 원본 추론용 checkpoint 바이트 그대로 |
| `predict.py` | 단일 CNN 독립 추론 코드 |
| `requirements.txt` | 확인한 패키지 버전 |
| `metadata.json` | 집계 지표·설정·시간·선택 epoch·코드 및 모델 해시 |
| `README.md` | 해당 run 사용법과 해석 범위 |
| `SHA256SUMS.json` | 나머지 다섯 파일의 SHA256 |

RGB/bilinear 224 letterbox, 고정 normalization0.5/0.5, Present=1/Absent=0, threshold0.5를 사용합니다. 추론에는 증강을 적용하지 않습니다.

원본 데이터·라벨·환자 ID·표본별 결과·분할표·과제 PDF·영상 포함 노트북·optimizer/scheduler/RNG 재개 state는 포함하지 않습니다. 로컬 `last_training_state.pt`와 배포용 best `model.pt`는 다릅니다. 배포 ZIP만으로 optimizer/epoch를 복원할 수 없습니다.

[준비 검사](preparation_checks.json)는 합성 fixture 점검이며 실제 모델 검증과 구분합니다. [실제 ZIP 검증 기록](package_validation.json)은 모든8개 모델의 원본 가중치 일치·내부 체크섬·합성영상 CPU 추론 및 실제 CLI 일치·기존 출력 보존을 확인합니다. 합성영상 점검을 과제 성능 평가로 해석하지 않습니다. 학습 코드·같은 환경의 epoch 경계 재개 방법은 [Task2 README](../../task2/README.md)를 참고하세요.

## 원본 로컬 기록에서 새 패키지 빌드

출력은 저장소 밖 새 경로여야 합니다. builder는 controller 완료·실제8개 검증·고정 설정·코드 및 checkpoint 해시를 확인하며 GitHub 게시나 최종 모델 선택을 수행하지 않습니다.

```bash
python tools/build_task2_training_settings_packages.py \
  --source-task2 '/path/to/original/task2' \
  --output '/path/outside/repository/task2-training-settings-20261124-v1'
```
