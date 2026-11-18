# Task2 후속 실험 — 2026-11-18

기록 상태: **전체5개 학습·저장 checkpoint 독립 검증·다운로드 패키지 검사 완료**. 다운로드 링크는 아래 이름의 GitHub Release가 공개된 뒤 동작한다.

완료 5개, 총 학습 43.77분. [지표 원본](metrics_summary.json)·[학습 기록](run_records/)·[모델 다운로드 레지스트리](../../models/task2-followups-20261118/manifest.json)·[학습 코드](../../task2/README.md). 모델 Release 태그는 `task2-followups-20261118-v1`이며 모델 링크는 Release 공개 뒤 동작한다.

| 모델/조건 | BA | 양성 recall | 음성 recall | best/완료 epoch | 학습 분 | peak MiB | 최저 환자 BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| GoogLeNet | 90.03% | 98.80% | 81.25% | 11/18 | 7.77 | 938 | 50.00% |
| VGG11 | 50.00% | 0.00% | 100.00% | 8/15 | 11.01 | 2819 | 50.00% |
| ResNet18/320/fold0 | 89.57% | 95.81% | 83.33% | 7/14 | 8.07 | 929 | 50.00% |
| VGG11/lr1e-4 | 80.21% | 85.63% | 74.79% | 4/11 | 8.09 | 2819 | 44.61% |
| VGG11/lr1e-4/fold1 | 76.28% | 82.98% | 69.58% | 5/12 | 8.83 | 2819 | 50.00% |

모든 값은 실제 측정값이다. 학습 시간은 epoch 루프·state 저장 포함이며 모델 구축/독립 검증 시간은 제외한다. 메모리는 epoch별 peak allocated 최대값이며 reserved/전체 GPU 점유와 다르다. CM은 각 run의 best_metrics.json에 있다. 검증 근거는 [CPU 합성 exact 재개 검사](resume_tests.json)와 [5개 checkpoint의 SHA256에 연결한 재로딩 검증](release_validation.json)에 있다. CPU 검증은 GPU bitwise 재현을 보장하지 않는다.

## 기존224와 해상도 후보

- fold0/seed42: 224 90.47% →320 89.57% (-0.90%p), 시간 1.47×; 환자 개선/동률/악화 1/4/1. 기존 최저 BA 환자 50.00%→50.00%.

두 fold 평균 +1%p·각 fold 무악화 기준을 충족하지 못했거나 확인이 부족해 기존224 정책을 유지한다. 사전 fold0 진입 gate 미충족으로 fold1 후보는 실행하지 않았다.

**VGG 조건부 학습률 진단:** 기본 lr3e-4의 최대 train BA 50.36%/validation BA 50.00%에서 미학습이 관측돼 학습률만1e-4로 바꾼 fresh run을1개 추가했다. 최대 train BA 100.00%, validation BA 80.21% (+30.21%p), 학습시간 0.73×. 같은 fold0/seed42에서의 제한된 진단이며 개선 안정성이나 최적 학습률을 주장하지 않는다.

**VGG lr1e-4 레시피의 추가 fold 확인:** 측정 2개 fold의 BA 단순 평균 78.25%, sample SD 2.78%p, 고유 검증 12환자·1315장이다. 같은 seed와 선택 checkpoint의 기술통계이며 train 환자가 겹친다. 학습률 matched 비교는 fold0 한 쌍뿐이고 fold1 lr3e-4 VGG 대조군은 없다. 두 fold 평균 학습률 개선·독립 Test·다른 seed 안정성으로 해석하지 않는다. fold0에서 기존 ResNet224와 비교한 환자 BA 개선/동률/악화는 1/2/3다. 최대 악화 환자의 BA는 100.00%→50.45% (-49.55%p)다. fold1에서 기존 ResNet224와 비교한 환자 BA 개선/동률/악화는 1/0/5다. 최대 악화 환자의 BA는 99.38%→53.75% (-45.63%p)다. 이 환자 비교는 구조·학습률이 다른 모델 간 보조 기술 비교이며 인과 ablation이 아니다.

[VGG 내부 진단](vgg_failure_diagnostic.json)은 기본 lr3e-4 모델을 고정한 training-only 8장(클래스별4장)의 eval 관측이다. 마지막 feature가 거의0이며 이미지 간 거의 상수인 출력이 확인됐으나 작은 표본이고 학습률이 원인이라고 확정하지 않는다. 파라미터 업데이트·새 라벨·환자별 표본 공개 없이 집계값만 보존했다.

ResNet224/320 비교에서는 입력 크기 외 seed·fold·batch·최대 학습 예산·checkpoint 선택 절차를 유지했다. 보간·증강 격자도 달라져 설정 전체의 효과다. 동일 환자를 여러 seed/fold에서 개발에 사용한 validation이며 독립 private Test로 부르지 않는다.

## 비교 범위와 규칙

기본 GoogLeNet/VGG11은 각각 첫 fold0/seed42 전체 학습이다. 한 조건만의 모델 순위는 안정성 증거가 부족하다. 각 모델의 무증강 대조군이 없어 geometry 효과를 해당 모델에서 입증했다고 하지 않는다. 환자별 BA는 공식 이미지 단위 BA의 보조 진단이며 단일 클래스 환자 BA는 정의하지 않는다.

제공 데이터만, random init/기본 지정 CNN, 독립 단일 모델, train-only 수평반전 p=.5와 ±7° 회전, 고정 RGB letterbox/.5 정규화, weighted CE, threshold .5를 유지했다. 사전학습·구조 변경·광도/ROI·추가 annotation·threshold/TTA·BN 보정은 미확정 상태로 미적용이다. GoogLeNet의 두 auxiliary loss는 각각 .3이다. GoogLeNet/VGG11 CUDA 학습에서 adaptive-pooling backward 비결정성 경고가 관측됐으며, CPU 합성 exact 재개 테스트와 구분하고 GPU bitwise 재현을 보장하지 않는다.

## 팀 재실행과 남은 작업

`config_recorded.json`은 당시 상대 경로/설정이고 `config.json`은 팀 저장소의 `data/task2`, `runs/task2` 경로를 쓰는 portable 설정이다. 데이터는 과제의 허가된 경로로 별도 준비하고 `task2/requirements.txt`의 환경을 사용한다. 예를 들어 저장소 루트에서:

```bash
python task2/code/googlenet.py --config experiments/task2-followups-20261118/run_records/googlenet_20261118_geometry_f0_s42/config.json --device cuda --run-name my_googlenet_f0
```

출력 이름은 새 이름으로 정한다. 새 full run의 source hash는 각 run_records/source_hashes.json에 보존했다. 코드가 변경된 미래 실행을 과거 결과의 정확 재현이라고 단정하지 않는다. 선택된 best 모델은 inference 가중치이고 optimizer/scheduler/RNG는 없다. 별도 last state는 로컬 epoch 경계 재개용이며 공개 패키지에서 제외했다. 중간 batch 재개·GoogLeNet/VGG11 GPU 수치의 완전한 일치는 보장하지 않는다.

전체 제공 데이터 최종 모델, 정식 채점 인터페이스 확인과 최종 제출은 미완료다. 전체 full run 최대5개 예산에서 실행하지 않은320 fold1의 자리1개를 VGG lr1e-4 fold0가 사전 gate를 통과한 뒤 동일 레시피 fold1 확인에 재배정했다. 총5개 full run을 완료했으며 새 튜닝 예산은 추가하지 않았다. 기존 Task1/7개 ResNet 공개 기록은 수정하지 않았다. 원본 영상·과제 문서·환자별 목록·개별 예측·학습 state는 이 공유 기록에 포함하지 않았다.
