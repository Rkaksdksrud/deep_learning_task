# 초기 ResNet18 baseline 원형 · 2026-11-01

기존 28개 실험 목록과 별도로 보존된 초기 full run 1개다. BA **47.5536%**, 최적2epoch/종료9epoch, 실측216.043초다. 환자 단위 개발 validation 결과이며 최종 Test·전체 데이터 제출 모델이 아니다. 원래 학습 source snapshot은 남아 있지 않다.

기존 독립 재검증을 통과했고 이번 보존 검사에서는 11/11 무증강 control과 모든122개 tensor·저장 예측·분할 bytes가 일치했다. 두 checkpoint 파일 bytes와 SHA는 서로 다르다. 당시 validation JSON에는 checkpoint SHA가 없으며 이번 내보내기에서 legacy를 새로 full 재추론한 것으로 표현하지 않는다. [보존 검사](preservation_checks.json)와 [실제 패키지 검사](package_validation.json)의 범위를 구분한다.

이 1개는 신규8개와 같은 `task2-augmentation-lr-20261125-v1` Release의 별도 ZIP으로 제공한다. [전용 manifest](manifest.json)의 원본/공개 checkpoint SHA·ZIP 크기와 SHA를 확인할 수 있다. 공개 복사본은 config의 `data_root`/`output_root` 두 값만 팀 경로로 바꾸며 가중치와 그 외 metadata는 동일하다. 원본 로컬 모델은 보존했다.

저장소 루트에서:

```bash
python models/task2-legacy-baseline-20261101/download.py --list
python models/task2-legacy-baseline-20261101/download.py --run resnet18_baseline_20261101 --output downloads/task2/original-baseline
python -m pip install -r downloads/task2/original-baseline/requirements.txt
python downloads/task2/original-baseline/predict.py --images input_images --output predictions.csv --device cpu
```

ZIP은 `model.pt`, 독립 `predict.py`, `requirements.txt`, 집계 `metadata.json`, `README.md`, `SHA256SUMS.json`의 여섯 flat 파일이다. 다운로더는 ZIP·내부 파일·가중치 SHA를 검증하며 기존 출력은 덮어쓰지 않는다. 개별 모델 하나로 예측하고 다른 모델과 평균·투표하지 않는다. threshold는0.5다.

원본 데이터·라벨·환자/표본 목록·분할표·개별 예측·학습 재개 state는 포함하지 않는다. 기능 점검용 smoke 모델은 full 실험 성과와 배포 모델에서 제외했다.

새 경로에 다시 패키징하려면 다음을 사용한다. 네트워크나 학습을 시작하지 않으며 이미 존재하는 출력·manifest는 보존한다.

```bash
python tools/build_task2_legacy_baseline_package.py --source-task2 /path/to/private/task2 --output /path/to/new/artifacts
```
