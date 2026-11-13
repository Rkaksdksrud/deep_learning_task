# Task2 ResNet18 — 7개 full try 모델 다운로드

Release tag: **task2-resnet18-20261111-v1** · [GitHub Releases](https://github.com/Rkaksdksrud/deep_learning_task/releases/tag/task2-resnet18-20261111-v1)

이번 비교의 전체7개 run에서 선택한 최적 checkpoint를 각각 독립 패키지로 제공합니다.
가중치는 Git 저장소에 중복 commit하지 않습니다. 각 링크는 해당 Release 발행 후 이용할 수 있습니다.
원본 데이터·개별 영상 정답/예측·환자 분할표·영상 포함 notebook은 패키지에 없습니다.

## 받기

Python 표준 라이브러리만으로 다운로드/무결성 확인이 가능합니다.

```bash
python models/task2-resnet18-20261111/download.py --list
python models/task2-resnet18-20261111/download.py --run geometry_f0_s42 --output downloads/task2/geometry_f0_s42
```

CPU 추론:

```bash
python -m pip install -r models/task2-resnet18-20261111/requirements.txt
python downloads/task2/geometry_f0_s42/predict.py --images input_images --output predictions.csv --device cpu
```

직접 받은 ZIP을 검사하고 압축을 풀 수도 있습니다.

```bash
python models/task2-resnet18-20261111/download.py --run geometry_f0_s42 --archive geometry_f0_s42.zip --output downloads/task2/geometry_f0_s42
```

권장 참조 모델은 원래 중심 조건인 `geometry_f0_s42`입니다. 다른 모델과 평균/투표하지 않습니다.
학습 시에만 증강하며 추론은 RGB224 letterbox, 고정 .5/.5 정규화, Present 확률 threshold .5입니다.

## 7개 결과와 직접 다운로드

| Run ID | fold/seed | 선택epoch | Valid BA | Present recall | Absent recall | ZIP |
|---|---|---:|---:|---:|---:|---|
| control_f0_s42 | 0/42 | 2 | 47.55% | 88.02% | 7.08% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/control_f0_s42.zip) |
| plain_ce_f0_s42 | 0/42 | 20 | 59.65% | 63.47% | 55.83% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/plain_ce_f0_s42.zip) |
| geometry_f0_s42 | 0/42 | 6 | 90.47% | 97.60% | 83.33% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/geometry_f0_s42.zip) |
| control_f0_s43 | 0/43 | 2 | 60.11% | 52.10% | 68.13% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/control_f0_s43.zip) |
| geometry_f0_s43 | 0/43 | 8 | 90.17% | 97.01% | 83.33% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/geometry_f0_s43.zip) |
| control_f1_s42 | 1/42 | 1 | 65.05% | 49.47% | 80.63% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/control_f1_s42.zip) |
| geometry_f1_s42 | 1/42 | 7 | 88.76% | 85.64% | 91.88% | [다운로드](https://github.com/Rkaksdksrud/deep_learning_task/releases/download/task2-resnet18-20261111-v1/geometry_f1_s42.zip) |

정확한 값·ZIP/가중치 SHA256·파일 크기는 [manifest.json](manifest.json)이 기준입니다.
ZIP 내부는 `resnet.pt`, `predict.py`, `requirements.txt`, `metadata.json`, `README.md`, `SHA256SUMS.json`입니다.

## 해석과 재현 범위

- 7개는 모두 원래 완료한 full run이며 smoke 가중치나 epoch별 중간 가중치는 포함하지 않습니다.
- geometry는 반전/회전/보간·크기 변화가 묶인 정책입니다. 개별 변환의 효과는 미분리입니다.
- 평균 BA89.80%는 두 환자 fold·세 조건의 개발 검증 결과입니다. 독립3-fold, 외부 검증 또는 private Test 결과가 아닙니다.
- 특정 검증 환자에서 BA50%,26.25% 실패가 남아 있습니다. 전체 데이터로 최종 재학습한 제출 모델도 아닙니다.
- checkpoint는 모델 상태/선택epoch metadata를 담지만 optimizer/scheduler/RNG 이어학습 상태는 없습니다.
- 허가받은 데이터를 별도 보유한 팀원은 [학습·평가 코드](../../task2/README.md)로 partition을 재생성해 aggregate 지표를 확인할 수 있습니다.
  원래 표본별 확률을 포함하지 않으므로 과거 saved-probability parity를 재확인하는 기능과 구분합니다.
- [실험 기록](../../experiments/task2-resnet18-20261111/README.md)을 함께 참고하세요.

## 배포 패키지 검증

2026-11-13에 7개 ZIP의 전체 파일/가중치 해시를 확인했습니다. 각 ZIP의 독립 추론 코드와 원래 코드에서 합성 RGB·회색·RGBA 영상 3개의 전처리 텐서·CPU 확률·이진 예측이 정확히 일치했습니다. 변조 ZIP 거절과 기존 출력 보존도 확인했습니다. 이는 패키징 검사이며 새 학습이나 holdout 재평가는 아닙니다. [검증 기록](package_validation.json)을 참고하세요.

## 배포 담당자

[tools/build_task2_packages.py](../../tools/build_task2_packages.py)는 원본 run에서 ZIP과 manifest를 만들고,
`tools/publish_task2_release.py`는 로컬 아티팩트 검사 후 GitHub Release에 올리는 용도입니다.
쓰기 인증이 필요하며 인증 토큰은 코드·README·Git commit에 넣지 않습니다.
