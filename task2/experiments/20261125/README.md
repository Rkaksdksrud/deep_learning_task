# 11월25일 실험의 코드와 재실행 설정

실제 학습한 ResNet18 코드와 환자 분할·평가·재개 구현을 보관한 버전이다. 기존 `task2/code/`는 바꾸지 않았다. 결과와 채택 판단은 [실험 보고서](../../../experiments/task2-augmentation-lr-20261125/README.md), 모델 다운로드는 [새 8개 모델](../../../models/task2-augmentation-lr-20261125/README.md)을 참고한다.

`code/`의 범용 Python 파일 10개는 실제 학습 workspace와 byte 단위로 같다. 환자 식별자가 들어 있는 로컬 진단 스크립트 3개는 제외했다. 원래 13개와 공개 10개의 SHA256, config의 경로 변경은 [source_export.json](source_export.json)에 남겼다. 기본 지정 CNN·무작위 초기화·고정 정규화·환자 분할·추론 임계값 0.5를 유지한다.

## 개별 실험 실행

저장소 루트에서, `task2/requirements.txt` 환경과 수업에서 허가받은 로컬 `data/task2/`를 준비한 뒤 실행한다. 원본 데이터·라벨·분할표는 공유하지 않는다.

```bash
# 먼저 실행 중인 학습이 없는지 확인하고, 새 run 이름을 사용한다.
python task2/experiments/20261125/code/resnet.py --config task2/experiments/20261125/configs/rotation_f0_s42.json --run-name my_rotation_f0_s42

# 자체적으로 새로 학습한 run의 epoch 경계 상태에서만 재개한다.
python task2/experiments/20261125/code/resnet.py --config task2/experiments/20261125/configs/rotation_f0_s42.json --resume runs/task2/20261125/my_rotation_f0_s42/last_training_state.pt
```

새 학습 config의 상대 경로는 `task2/experiments/20261125/` 기준이다. 따라서 `../../../data/task2`와 `../../../runs/task2/20261125`는 저장소 루트의 데이터와 새 출력 위치를 가리킨다. CLI의 `--config`, `--resume` 경로는 현재 터미널 기준이다. 기존 출력은 덮어쓰지 않는다.

`configs/`에는 full 비교용 10개와 기능 확인용 `smoke_*.json` 4개가 있다. 실제 full 학습은 회전 3개·무증강 LR 1개·밝기 3개·대비 1개, 총 8개를 완료했다. 대비 추가 2개는 사전 기준 미충족으로 실행하지 않았다. smoke는 학습 성능 증거가 아니다. 작은 동작 확인은 기존 CLI의 `--smoke-test`를 사용할 수 있다.

학습 중 생성되는 `last_training_state.pt`에는 optimizer·scheduler·epoch·RNG가 포함된다. 배포 ZIP의 `model.pt`는 추론용이므로 그 ZIP만으로 중간 학습을 재개할 수 없다. 공개 source/config는 경로가 달라 원본 로컬 state를 그대로 재개하기 위한 파일이 아니며, 본 환경에서 새로 시작한 run을 재개하는 용도다. batch 중간 복원이나 GPU bitwise 재현은 보장하지 않는다.

밝기·대비는 기존 기하 증강에 각각 별도로 추가한 탐색 실험이다. 교수자 허용 여부는 미확정이며 제출 설정에 채택하지 않았다. 여러 모델의 평균·투표도 사용하지 않는다.

## 임계값의 조건부 진단

`analysis/threshold_analysis.py`와 합성 검사는 실제 분석 원본과 byte 단위로 같다. 원본 스크립트의 기본 CLI는 비공개 학습 기록·SHA-bound 예측 파일을 요구하므로 공유 저장소만으로 실행할 수 없다. 그 검증을 건너뛴 결과를 원본 성능 재현이라고 표현하지 않는다.

별도 어댑터는 사용자가 로컬에 가진, 출처와 분할을 확인한 예측 CSV에 동일한 LOPO 알고리즘을 적용한다.

```bash
python -m unittest discover -s task2/experiments/20261125/analysis -p test_threshold_analysis.py
python task2/experiments/20261125/analysis/threshold_lopo.py --predictions /path/to/private_predictions.csv --output /path/to/new_private_diagnostic.json
```

CSV에는 `patient_group,label,probability`가 필요하다. 이 어댑터는 checkpoint·분할 provenance를 검증하지 않으며 배포용 threshold를 선택하지 않는다. checkpoint epoch가 전체 validation에서 선택되었다면 LOPO도 그 의존성을 제거하지 못한다. 출력에 환자별 기록이 있으므로 로컬에 보관한다. 공개 결과는 [환자 식별자를 제외한 요약](../../../experiments/task2-augmentation-lr-20261125/threshold_summary.json)이다.
