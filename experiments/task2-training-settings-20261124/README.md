# ResNet18 학습 설정 단일 변경 비교 — 2026-11-24

기록 상태: **학습·독립검증 완료 / 공유 기록 준비**. push·Release 공개 성공은 이 exporter가 확인하지 않는다.

사용자 선택에 따라 **수평반전·회전 등 모든 학습 증강을 껐다**. 공통 reference는 ResNet18/random init, weighted CE·lr3e-4·WD1e-4·cosine이며 fold0/seed42 BA 47.55%다. 기존 geometry 모델의90%대 점수와 비교한 개선이 아니다. 실제 신규 full run 8개, 학습 합계 47.13분.

| 축/변경 | fold0/seed42 BA | reference 대비 차이 | 확인 조건 수 | matched 평균 차이 | 해석 |
|---|---:|---:|---:|---:|---|
| 학습 클래스 가중치 (weighted CE → 일반 CE) | 59.65% | +12.10%p | 3 | +6.68%p | 조건에 따라 방향이 바뀜; 현재 reference 유지 |
| 학습률 (3e-4 → 1e-4) | 56.44% | +8.89%p | 2 | +7.56%p | 두 개발 fold 상승; 추가 seed·환자 악화 확인 전 채택 보류 |
| Weight decay (1e-4 → 0) | 53.94% | +6.39%p | 1 | +6.39%p | fold0 단발 탐색 결과; 일반적 우열·안정성 미확인 |
| 스케줄 (cosine → constant) | 53.89% | +6.33%p | 1 | +6.33%p | fold0 단발 탐색 결과; 일반적 우열·안정성 미확인 |

[측정 JSON](metrics_summary.json)·[matched 비교 CSV](comparison.csv)·[실행별 기록](run_records/)·[모델 다운로드 레지스트리](../../models/task2-training-settings-20261124/manifest.json)·[학습 코드](../../task2/README.md). 모델 Release 태그는 `task2-training-settings-20261124-v1`이며 다운로드는 실제 공개 뒤 가능하다.

## 비교 해석과 선택 절차

각 후보는 다른 설정 세 개를 유지한 채 하나만 바꾼다. 독립·가산적인 기법별 기여나 전 모델에 통하는 최적값을 찾은 실험이 아니다. 일반 CE 후보도 학습 가중치만 제거하며 validation loss와 BA 동률 checkpoint 선택은 Train 수로 계산한 동일 weighted CE다. 수치가 다른 학습 loss의 크기 자체를 성능 순위로 쓰지 않는다. LR은30epoch/patience7 안에서의 수렴 효과까지, constant는 누적 LR 노출·조기종료 경로까지 포함한다. WD0은현재1e-4 제거만 검증한다.

최대8개 full run 예산에서 기본5개를 수행한 뒤 fold0 차이 절대값1%p 이상인 최대2축만 fold1에서 확인한다. 두 fold 방향이 같고 평균 차이 절대값1%p 이상인 축 중1개만 seed43에서 확인하며 안정적인 상승 후보를 우선한다. 큰 하락도 현재 값의 유용성을 검토하기 위해 확인 대상으로 삼았다. 축별 확인 예산이 다르고 후보 선택 편향이 남는다. 숫자 채택 gate는 세 matched 조건 평균 +1%p·최소2향상·어느 조건도-1%p 초과 하락 없음이며, 환자별 악화·두 recall·비용을 별도로 검토해야 한다. 한 번의 상승만으로 교체하지 않는다.

일반 CE의 세 조건 평균 차이는 +6.68%p지만 seed43 차이가 -5.30%p여서 사전 채택 gate를 통과하지 못했다. 학습 클래스 가중치의 보편적인 필요/불필요를 결론내리지 않는다. 낮춘 LR은 두 개발 fold 평균 +7.56%p로 상승했지만 추가 seed가 없다. 특히 fold0 선택 모델의 양성 recall은 19.76%, 음성 recall은 93.12%, 양성 미검출은 134/167개다. BA 상승을 두 클래스의 안정적인 개선으로 해석하지 않는다. 축별 확인 조건 수가 달라 각 평균을 공정한 최종 순위로 바로 비교하지 않는다. 모든 run에서 관측 train BA 최대값은100%였지만 validation과 격차가 남았다. 과적합·환자 간 분포 차이와 양립하며 원인을 분리한 진단은 아니다. 최대30epoch/patience7 규칙을 유지했고 더 오래 학습하면 해결된다고 결론내리지 않는다.

모든 fold는 이미 개발에 사용한 patient holdout validation이다. 같은 fold0의 seed 반복은 독립3-fold가 아니며 private Test도 아니다. 환자 BA는 보조 진단이고 한 클래스만 있는 환자는 BA를 정의하지 않는다. 공개 JSON에는 환자 ID 없이 개선/악화 개수와 최저 BA만 담았다. 시간 배율은 실제 종료 epoch·저장 I/O가 포함된 전체 run 값이며, 과거 대조군은 매epoch last-state 저장이 없으므로 순수 연산비용 효과가 아니다.

## 검증 근거와 규칙

[CPU17개 정책·재개·평가 가중치 검사](scheduler_and_eval_tests.json), [새/과거 reference parity](baseline_parity.json), [과거 대조군의 새 추론 audit](historical_control_audits.json), 실행별 validation_checks.json과 checkpoint SHA256을 참고한다. 합성 CPU 검증은 실제 과제 성능이나 GPU bitwise 학습 재현을 뜻하지 않는다. 제공 데이터만, 기본 지정 ResNet18, RGB224 letterbox/고정 .5 정규화, batch16 FP32 AdamW, 단일 모델/threshold .5를 유지했다. 사전학습·추가 라벨·새 전처리 신경망·구조 변경·앙상블·TTA·임계값 튜닝은 적용하지 않았다. 기존12개 전체 실험·Task1·원본을 보존했다. 영상·라벨·환자/표본 목록·개별 예측·로컬 notebook·학습 state는 공개 기록에서 제외했다.

## 팀 재실행

허가된 과제 데이터는 저장소의 `data/task2`에 별도 준비한다. `config_recorded.json`은 당시 상대 경로, `config.json`은 팀 저장소 경로를 쓰는 설정이다. 기존 환경과 `task2/requirements.txt`를 사용하고 저장소 루트에서:

```bash
python task2/code/resnet.py --config experiments/task2-training-settings-20261124/run_records/resnet18_20261124_reference_f0_s42/config.json --device cuda --run-name my_reference_f0
```

다른 후보는 해당 run_records의 config.json으로 바꾸고 새 run 이름을 쓴다. 다른 모델·증강·fold 조합으로 무제한 탐색하지 않는다. 코드 버전은 실행별 source_hashes.json에 있다. 선택된 best 가중치는 추론용이며 optimizer/scheduler/RNG가 없고, 별도 last state는 로컬 동일 환경의 epoch 경계 재개용이다. mid-batch 재개·GPU bitwise 일치를 주장하지 않는다. 이 결과는 전체 데이터 최종 모델이나 최종 제출 점수가 아니다.
