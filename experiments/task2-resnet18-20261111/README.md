# Task2 ResNet18 — 2026-11-11 비교 실험

7개 전체 학습과 체크포인트 독립검증을 완료한 기록이다. 합계 **95 epoch, 학습 루프 2,308.8초(38.48분)**이며, 2026-11-13에 팀 공유용으로 정리했다. 학습·추론 코드는 [task2](../../task2/), 모델별 Release ZIP 7개와 해시는 [모델 manifest](../../models/task2-resnet18-20261111/manifest.json)를 따른다. Release 태그는 `task2-resnet18-20261111-v1`, 각 파일명은 `<run_id>.zip`이다.

기존 자료의 확인 범위·채택/제외 이유와 미실행 후속 계획은 [자료 검토와 다음 시도](RESEARCH_AND_NEXT_STEPS.md)에 정리했다.

현재 ResNet18 레시피로 **학습에만 수평반전 p=0.5 + ±7° 회전 정책**을 채택했다. 같은 fold·seed 대조군과 비교한 3조건에서 BA가 모두 상승했다. 참조 체크포인트는 단일 `geometry_f0_s42`이며, 7개 모델은 각각 별도 비교용 모델이다. 전체 제공 데이터로 재학습한 최종 제출 모델은 아직 없다.

| 환자 fold / 학습 seed | Control BA | Geometry BA | 차이 | Geometry Present recall | Geometry Absent recall |
|---|---:|---:|---:|---:|---:|
| fold0 / 42 | 47.55% | 90.47% | +42.92%p | 97.60% | 83.33% |
| fold0 / 43 | 60.11% | 90.17% | +30.06%p | 97.01% | 83.33% |
| fold1 / 42 | 65.05% | 88.76% | +23.71%p | 85.64% | 91.88% |
| 세 조건 단순 평균 | 57.57% | 89.80% | +32.23%p | 93.42% | 86.18% |

Geometry BA의 조건 간 표본 SD는 0.91%p, Control은 9.02%p, paired 차이의 SD는 9.78%p다. 이 3조건은 **독립 3-fold 교차검증이나 private Test 결과가 아니다.** fold0의 같은 647장·6환자를 두 seed에서 반복했고 fold1은 668장·다른 6환자다. 고유 검증 대상은 1,315장·12환자이며, 검증 점수는 체크포인트와 후보 선택에도 사용됐다. SD는 기술통계이며 신뢰구간이 아니다.

일반 CE는 fold0/seed42 한 조건에서 BA 59.65%(Control 대비 +12.10%p), Present recall 63.47%, Absent recall 55.83%였다. 사전 예산에 따라 초기 개선이 더 큰 Geometry만 추가 확인했으므로 **일반 CE는 미채택·안정성 미확인**이다. 다른 조건에서 효과가 없다고 입증한 결과는 아니다. 두 후보의 결합은 실험하지 않았다.

## 전체 실행 기록

| run_id | 완료 / 최적 epoch | BA | 학습초 | 상태 |
|---|---:|---:|---:|---|
| control_f0_s42 | 9 / 2 | 47.55% | 210.71 | 검증 완료 |
| plain_ce_f0_s42 | 27 / 20 | 59.65% | 642.66 | 검증 완료 |
| geometry_f0_s42 | 13 / 6 | 90.47% | 329.42 | 검증 완료 |
| control_f0_s43 | 9 / 2 | 60.11% | 213.51 | 검증 완료 |
| geometry_f0_s43 | 15 / 8 | 90.17% | 376.46 | 검증 완료 |
| control_f1_s42 | 8 / 1 | 65.05% | 184.76 | 검증 완료 |
| geometry_f1_s42 | 14 / 7 | 88.76% | 351.27 | 검증 완료 |

[leaderboard.csv](leaderboard.csv)는 기록을 찾기 위한 Valid BA 내림차순 표다. fold가 다른 결과의 순위를 최종 모델 우열로 해석하지 않는다. 같은 조건 비교는 [paired_geometry.csv](paired_geometry.csv), 평균·변동·비용과 범위는 [metrics_summary.json](metrics_summary.json)에 있다. 모든 CSV의 BA/재현율은 0–1 비율이고 차이에 100을 곱하면 %p다. `elapsed_seconds`는 전체 학습 루프 시간이며 별도 검증·공유 준비 시간은 제외한다.

## 데이터와 비교 조건

제공 영상 2,948장, 익명 환자 27명, RGB PNG 480×724(W×H), Absent 2,160장·Present 788장이었다. 클래스는 0=Absent, 1=Present다. 환자 구분은 제공 익명 환자 열을 사용했다. 원본 연결·손상·정확한 파일/픽셀 중복 감사는 통과했으며 같은 환자에 유사 프레임이 많아 환자 단위로 분리했다.

`StratifiedGroupKFold(5, shuffle=False)`의 지정 fold만 Valid로 사용했다. fold0은 Train 2,301장/21환자·Valid 647장/6환자, fold1은 Train 2,280장/21환자·Valid 668장/6환자다. 각 run의 Train/Valid 환자는 겹치지 않는다. 분할 구성은 결과를 본 뒤 변경하지 않았다.

공통 설정은 random initialization(`weights=None`)의 기본 ResNet18·2-class head, 224 RGB letterbox, 고정 `(pixel/255 - 0.5)/0.5`, AdamW(lr=0.0003, weight decay=0.0001), batch 16, FP32, cosine schedule, 최대 30 epoch, patience 7, threshold 0.5다. 클래스 가중치는 Train 클래스 수로만 계산했다. 최적 epoch는 Valid BA 최대, 동률이면 Valid loss 최소로 선택했다. 일반 CE는 손실 가중치만 바꾸고, Geometry는 Train 기하 정책만 바꿨다.

Geometry는 실제 영상 영역을 줄인 뒤 수평반전·`expand=True` 소회전·최종 letterbox를 적용하며 Valid/추론 변환은 그대로다. **반전·회전과 이에 수반되는 보간·크기 변화 정책 전체의 효과**를 측정했다. 개별 반전이나 회전의 기여를 분리한 실험이나 논문 성능 재현은 아니다. Geometry 학습시간 평균은 352.39초, Control은 202.99초(합계 시간 비율 1.736배)였다. 같은 최대 예산과 조기종료 규칙에서 실제 완료 epoch가 달랐으므로 순수 증강 연산 비용과 추가 수렴 epoch가 섞여 있다.

## 남은 실패와 한계

fold0의 한 실패 환자는 Geometry 두 seed 모두 99장 전부 Present로 예측해 Absent 80장을 모두 오탐했고 환자 BA가 **50%**였다. fold1의 한 실패 환자는 Geometry BA **26.25%**로 Control 35.625%보다 **9.375%p 악화**했다. Present 27장을 모두 놓치고 Absent 80장 중 38장을 오탐했다. 해당 fold의 FN 27개 전부와 FP 39개 중 38개가 이 환자에 집중됐다.

18개 환자×조건 관측은 15개 개선·2개 동률·1개 악화였으며, 같은 fold0 환자를 두 seed에서 반복한 수치다. 높은 전체 평균이 모든 새 환자의 안정성을 뜻하지 않는다. 환자별 임계값·라벨 변경·실패 환자 제외·private Test 재선택은 사용하지 않았다.

외부 데이터·사전학습 가중치·추가 라벨·다른 신경망 전처리·앙상블은 사용하지 않았다. 사전학습, 구조 변경, 광도 증강, 임계값/TTA 등 허용 범위가 확인되지 않은 기능은 보류했다. VGG·GoogLeNet의 전체 학습, 전체 데이터 최종 학습과 전체 과제 제출물은 아직 미완료다.

## 파일 해석과 재사용

- `run_records/<run_id>/history.csv`, `best_metrics.json`: 실제 기록한 epoch별 집계와 선택 epoch 지표다.
- `config_recorded.json`: 기록 당시 설정을 그대로 보존했다. 경로 `data/raw`, `runs`는 당시 코드의 Task2 폴더 기준 상대값이다.
- `config.json`: 공유 코드에 맞춰 `data_root=../data/task2`, `output_root=../runs/task2`로 정리했다. 두 경로는 [task2](../../task2/) 기준이다. 원본에 생략됐던 `geometric_augmentation=false` 기본값만 명시했고 학습 수치·하이퍼파라미터는 바꾸지 않았다.
- `training_summary.json`: epoch·시간·BA는 유지하고 로컬 `weights_path`를 제거해 모델 manifest/run ID/Release asset 참조로 바꿨다. 이 참조의 경로 기준은 저장소 루트다.
- `environment.json`, [environment_versions.json](environment_versions.json): 실제 기록한 패키지·Python/CUDA/cuDNN·GPU 버전만 남겼다. Python 실행 파일의 머신 경로는 제거했다.
- [source_hashes.json](source_hashes.json): 원래 학습 소스 스냅샷의 코드 SHA-256이다. 최초 Control은 자동 manifest 도입 전 참조 스냅샷이며, 나머지 6개는 run 시작 시 manifest와 스냅샷 byte를 대조했다. 공유용 코드의 경로 변경으로 현재 [task2](../../task2/) 코드와 byte가 다를 수 있다.
- [validation_checks.json](validation_checks.json): 7개 run의 기존 독립검증 통과와 저장 지표/history 일관성 확인 기록이다. 공유 준비 중 새 학습·추가 holdout 평가는 하지 않았다. 합성입력 기반 배포검사는 [package_validation.json](../../models/task2-resnet18-20261111/package_validation.json)을 참고한다.

재학습·모델 다운로드·추론 명령은 [task2 README](../../task2/README.md)와 [모델 manifest](../../models/task2-resnet18-20261111/manifest.json)를 따른다. 이 폴더에는 가중치 복사본, 원본 데이터·과제 PDF·환자/프레임 분할 명단·개별 영상 정답/예측·영상·이미지가 든 노트북을 포함하지 않는다.
