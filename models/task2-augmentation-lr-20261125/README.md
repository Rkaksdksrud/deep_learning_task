# Task2 회전·학습률·밝기/대비 모델 — 2026-11-25

완료·독립 검증한8개 holdout 모델의 패키지입니다. [manifest.json](manifest.json)에 원본/배포 가중치 및 ZIP 해시를 기록했습니다.
Release `task2-augmentation-lr-20261125-v1`을 공개했고 모든 원격 파일 크기·SHA256과 대표 모델의 로그인 없는 실제 다운로드를 확인했습니다. [게시 기록](../../experiments/task2-augmentation-lr-20261125/SHARING_STATUS.json)을 참고하세요. 기존20개 모델은 보존합니다.

| Run | 검증 BA | 규칙 상태 | 다운로드 |
|---|---:|---|---|
| resnet18_20261125_rotation_f0_s42 | 90.2083% | within_confirmed_scope | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_rotation_f0_s42.zip) |
| resnet18_20261125_lr1e4_f0_s43 | 63.9371% | within_confirmed_scope | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_lr1e4_f0_s43.zip) |
| resnet18_20261125_brightness_f0_s42 | 91.8114% | unconfirmed_exploratory_only | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_brightness_f0_s42.zip) |
| resnet18_20261125_contrast_f0_s42 | 91.4583% | unconfirmed_exploratory_only | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_contrast_f0_s42.zip) |
| resnet18_20261125_rotation_f0_s43 | 87.4757% | within_confirmed_scope | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_rotation_f0_s43.zip) |
| resnet18_20261125_rotation_f1_s42 | 87.3825% | within_confirmed_scope | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_rotation_f1_s42.zip) |
| resnet18_20261125_brightness_f1_s42 | 88.4597% | unconfirmed_exploratory_only | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_brightness_f1_s42.zip) |
| resnet18_20261125_brightness_f0_s43 | 88.0739% | unconfirmed_exploratory_only | [ZIP](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-augmentation-lr-20261125-v1/resnet18_20261125_brightness_f0_s43.zip) |

```bash
python models/task2-augmentation-lr-20261125/download.py --list
python models/task2-augmentation-lr-20261125/download.py --run resnet18_20261125_rotation_f0_s42 --output downloads/task2-augmentation-lr/rotation
python -m pip install -r downloads/task2-augmentation-lr/rotation/requirements.txt
python downloads/task2-augmentation-lr/rotation/predict.py --images input_images --output predictions.csv --device cpu
```

각 ZIP은 `model.pt,predict.py,requirements.txt,metadata.json,README.md,SHA256SUMS.json`의 여섯 flat 파일입니다.
다운로더는 ZIP과 내부 전체 파일의 SHA를 확인합니다. 여러 모델을 결합해 예측하지 않습니다.

원본 checkpoint의 머신 경로 유출을 막기 위해 배포 복사본의 config.data_root/output_root 두 값만 portable 경로로 변경해 재직렬화했습니다.
모든 model tensor와 다른 metadata·임계값0.5는 정확히 같고 로컬 원본 bytes는 보존했습니다. `original_checkpoint_sha256`와
`published_checkpoint_sha256`는 서로 다른 파일을 식별합니다. 실제 CPU/CLI 비교 범위는 `package_validation.json`을 확인하세요.

밝기·대비·threshold 허용 범위는 미확정입니다. 탐색 결과를 제출 설정에 자동 채택하지 않습니다.
이8개는 환자 holdout 모델이며 전체 데이터 최종 제출 모델이나 private Test 성능이 아닙니다.
원본 데이터·영상/라벨·환자목록·개별예측·분할표·과제PDF·학습 재개 state를 포함하지 않습니다.
비교/해석은 [실험 기록](../../experiments/task2-augmentation-lr-20261125/)을 참고하세요.
