# Task2 회전·학습률·밝기/대비·임계값 비교 — 2026-11-25

**학습·독립 검증 완료 / GitHub 모델 공유 준비 상태.** 신규8개 full run을 최대8개 사전 예산 안에서 GPU 한 작업씩 실행했다. 총137epoch, 실측 학습3,416.467451초(56.94분)다. 코드·config·집계 지표와 모든 실제 try의 모델을 공유 대상으로 준비하며, 이 README는 push나 Release 게시 성공을 확인한 기록이 아니다.

[집계 JSON](metrics_summary.json) · [matched 비교 CSV](comparison.csv) · [실행별 기록](run_records/) · [버전 코드와 재실행](../../task2/experiments/20261125/README.md) · [모델 manifest](../../models/task2-augmentation-lr-20261125/manifest.json) · [전체 실험 노트북](../../task2/notebooks/task2_experiment_overview.ipynb).

새 모델 묶음은 `task2-augmentation-lr-20261125-v1` 태그에 연결하며, 해당 Release가 실제 공개된 뒤 다운로드할 수 있다. 기존7+5+8개 모델과 이번8개, 별도 초기 원형1개를 합쳐 **29개 모델을 공유 관리**한다. 이번 Release에는 신규8개와 legacy1개 총9개 ZIP을 연결하며 [legacy 별도 manifest](../../models/task2-legacy-baseline-20261101/manifest.json)를 제공한다. 미실행 template에는 가중치나 성능을 부여하지 않는다.

## 초기 원형·smoke를 포함한 전체 분류

| 분류 | 개수 | 상태·공유 범위 |
|---|---:|---|
| 기존11/11·11/18·11/24 전체 실험 |20|기존 모델·기록 보존|
| 이번11/25 신규 full run |8|학습·독립 검증 완료, 새 모델 manifest로 공유 준비|
| 초기11/01 baseline 원형 |1|재검증 완료, legacy 별도 모델 manifest로 공유 준비|
| `smoke_geometry_20261111`, `smoke_resnet18_20261124_oat` |2|1epoch 기능 점검·검증 완료, 성과/모델 배포 제외|
| `smoke_resnet_check`, `smoke_vgg_check`, `smoke_googlenet_check` |3|smoke 표시와 best 가중치는 있으나 종료 summary/검증 기록 부족, 성과/모델 배포 제외|

[초기 baseline 원형](../../models/task2-legacy-baseline-20261101/README.md)은 `resnet18_baseline_20261101`이다. 실제9epoch/best2, BA47.5536%, CM `[[34,446],[20,147]]`, 학습216.043초다. 기존 독립 재검증은 저장 예측 불일치0으로 통과했다. 이번 CPU 검사에서11/11 재학습 대조군과 모든 tensor·저장 prediction CSV·split bytes가 같음을 확인했지만 checkpoint bytes는 다르다. 원본 SHA256은 `5e42a1b56ea37086742628c69770d31662a092aa40f78e34356defc9442c00d9`다.

당시 원래 source snapshot과 SHA-bound audit는 없어 provenance에 그 한계를 표시한다. 재검증 기록·현재 CPU tensor parity를 원래 소스 증명이나 새 일반화 실험으로 해석하지 않는다. legacy는 신규 학습8회·137epoch·56.94분 합계에 포함하지 않는다. smoke3개의 미완료 원인은 기록만으로 확정하지 않으며 실패했다고 단정하지 않는다. local run inventory34개를 위처럼 full29개+smoke5개로 구분했다.

## 무엇이 도움이 됐나

| 변경 | 같은 조건의 BA 차이 | 평균 차이 ± 표본 SD | 판단 |
|---|---|---:|---|
| 기존 flip+rotation → 회전 단독 | −0.2607 / −2.6940 / −1.3741%p | −1.4429 ± 1.2181%p | 세 조건 모두 하락, 기존 combined 정책 유지 |
| **무증강** LR3e-4 → 1e-4 | +8.8891 / +6.2367 / +3.8267%p | +6.3175 ± 2.5322%p | 세 조건 수치 gate 통과, 특정 환자 최대−33.75%p 악화 남음 |
| geometry + 밝기 | +1.3423 / −0.2970 / −2.0958%p | −0.3502 ± 1.7197%p | 첫 상승이 다른 fold/seed에서 유지되지 않음 |
| geometry + 대비 | fold0/seed42 +0.9893%p | 한 조건, SD 미정의 | 추가 확인 gate +1%p 미달, 추가2조건 미실행 |

회전 조건 순서는 fold0/seed42, fold0/seed43, fold1/seed42다. 밝기 조건 순서는 fold0/seed42, fold1/seed42, fold0/seed43이다. LR은 이전11/24의 fold0/seed42·fold1/seed42에 이번 fold0/seed43을 추가했다. LR의 당시 “추가 seed 미확인” 상태는 해소됐지만, **무증강 조건의 개선**이며 기존 geometry/LR3e-4 모델의 대체 근거는 아니다. geometry+LR1e-4 조합은 수행하지 않았다.

회전 단독은 현재 combined 정책에서 flip을 제거한 비교다. rotation 자체의 독립 기여나 flip-only와의 비교가 아니다. 세 조건 수치 gate는 평균+1%p 이상·최소2상승·어느 조건도−1%p 초과 하락 없음이며 환자 실패·두 recall·비용을 별도로 본다. SD는 조건 간 기술통계로 신뢰구간이 아니다. 같은 fold0 두 seed는 환자를 공유해 독립3-fold가 아니다.

**밝기/대비 증강과 threshold 변경은 공식 허용 여부가 미확정이다.** 사용자 요청에 따른 탐색이며 교수자 허용으로 해석하지 않고 제출 설정 채택에서 제외한다. 모든 새 학습 모델의 추론 threshold는.5다. 제공 데이터·원래 라벨·random-init 기본 ResNet18·단일 모델을 사용했고 외부 가중치·추가 라벨·추가 신경망·앙상블·TTA는 사용하지 않았다.

## 신규8개 실행 결과

BA·recall은 전체 validation의 이미지 단위다. CM 순서는 `[[TN,FP],[FN,TP]]`, epoch는 최적/종료다. 모든 모델은 저장 checkpoint를 독립 재로딩해 검증했다.

| run ID | BA | 양성 / 음성 recall | CM | epoch | 학습초 | peak MiB |
|---|---:|---|---|---:|---:|---:|
| resnet18_20261125_rotation_f0_s42 | 90.2083% | 100.0000% / 80.4167% | [[386, 94], [0, 167]] | 4/11 | 278.730 | 578.34 |
| resnet18_20261125_lr1e4_f0_s43 | 63.9371% | 62.8743% / 65.0000% | [[312, 168], [62, 105]] | 5/12 | 281.439 | 578.34 |
| resnet18_20261125_brightness_f0_s42 | 91.8114% | 88.6228% / 95.0000% | [[456, 24], [19, 148]] | 2/9 | 230.007 | 578.34 |
| resnet18_20261125_contrast_f0_s42 | 91.4583% | 100.0000% / 82.9167% | [[398, 82], [0, 167]] | 18/25 | 634.540 | 578.34 |
| resnet18_20261125_rotation_f0_s43 | 87.4757% | 86.8263% / 88.1250% | [[423, 57], [22, 145]] | 9/16 | 395.223 | 578.34 |
| resnet18_20261125_rotation_f1_s42 | 87.3825% | 78.7234% / 96.0417% | [[461, 19], [40, 148]] | 26/30 | 740.820 | 576.59 |
| resnet18_20261125_brightness_f1_s42 | 88.4597% | 77.1277% / 99.7917% | [[479, 1], [43, 145]] | 6/13 | 322.566 | 576.59 |
| resnet18_20261125_brightness_f0_s43 | 88.0739% | 92.8144% / 83.3333% | [[400, 80], [12, 155]] | 14/21 | 533.142 | 578.34 |

공통 설정은 SGKF5(shuffle=False)의 기존 patient split, 224 RGB/bilinear letterbox, 고정 normalization .5/.5, batch16 FP32 AdamW, WD1e-4, cosine max30/patience7이다. LR 추가 seed만1e-4·무증강이며 나머지는LR3e-4·geometry 기반이다. Train-count weighted CE와 동일 weighted validation loss를 유지하고 Valid BA 최대/동률 loss 최소로 checkpoint를 선택했다. 학습에서 추정하는 클래스 수는 Train만 사용하며 Valid/추론에는 증강이 없다.

밝기와 대비는 각각 독립 factor U[.85,1.15]를 Train 실영상 영역에 적용한 후 기하변환·검정 padding을 수행한다. 두 변환 결합·강도 탐색은 하지 않았다. 기존 flip decision RNG draw를 유지하고 회전 단독에서는 flip 적용만 껐다.

## 환자별 실패와 선택의 한계

| 신규 비교 | 환자 개선 / 동률 / 악화 | 최대 환자 BA 하락 |
|---|---:|---:|
| resnet18_20261125_rotation_f0_s42 | 1 / 4 / 1 | -6.6667%p |
| resnet18_20261125_lr1e4_f0_s43 | 3 / 2 / 1 | -33.7500%p |
| resnet18_20261125_brightness_f0_s42 | 3 / 1 / 2 | -21.4286%p |
| resnet18_20261125_contrast_f0_s42 | 2 / 4 / 0 | 하락 없음 |
| resnet18_20261125_rotation_f0_s43 | 1 / 0 / 5 | -35.8333%p |
| resnet18_20261125_rotation_f1_s42 | 1 / 1 / 4 | -50.0000%p |
| resnet18_20261125_brightness_f1_s42 | 1 / 2 / 3 | -50.0000%p |
| resnet18_20261125_brightness_f0_s43 | 1 / 4 / 1 | -17.7419%p |

LR 추가 seed는 전체BA가+3.8267%p 상승해도 한 환자가50.00→16.25%(−33.75%p)로 악화했다. 밝기·회전의 fold1에서는 각각 한 환자100→50%(−50%p) 악화가 있었다. 최저 환자 BA가50%로 올라간 경우에도 양성을 모두 놓치는 실패가 남을 수 있다. 공개본에는 환자 ID·표본 목록·개별 결과를 넣지 않으며 환자 BA는 공식 지표의 대체가 아닌 보조 진단이다. 한 클래스 환자는 BA를 정의하지 않고 관측 클래스 recall을 사용한다.

8개 모두 관측 train BA 최대는100%였다. validation 격차는 과적합·환자 간 분포 차이와 양립하지만 원인을 분리한 진단은 아니다. 초기 수렴만 보고 성공을 결정하지 않았다. 밝기 첫 조건+1%p 이상을 근거로 다른 fold·seed를 확인했으며, 두 photometric 후보 중 최대1축만 추가 확인한 선택 편향과 다른 튜닝 예산이 남는다. 대비 추가 fold1/seed42·fold0/seed43 두 template는 사전 gate로 미실행했다. 최종 Test·전체 데이터 최종 학습·제출은 수행하지 않았다.

시간은 조기종료 epoch와 재개 state 저장 I/O를 포함한 전체 run 값이다. 과거 일부 대조군에는 매epoch last-state 저장이 없어 시간 비율을 순수 augmentation/LR 연산 비용으로 해석하지 않는다.

## threshold: 별도의 조건부 LOPO 진단

기존 geometry checkpoint3개를 고정하고 다른5검증환자에서 pooled image-level BA를 최대화하는 전역 threshold를 고른 뒤 남은1환자에 적용했다. 사전 grid는.05~.95, step.01; 동률은.5에 가까운 값, 다음으로 작은 값이다. heldout 환자의 라벨과 확률은 그 환자의 threshold 선택에 사용하지 않는다. 이 절차는 환자별 threshold를 배포하는 정책이 아니며 **배포용 단일 threshold는 선택하지 않았다.**

| 고정 모델 조건 | fixed.5 BA | LOPO BA | 차이 | LOPO 양성 / 음성 recall | LOPO CM |
|---|---:|---:|---:|---|---|
| fold0/seed42 | 90.4691% | 91.3673% | +0.8982%p | 99.4012% / 83.3333% | [[400, 80], [1, 166]] |
| fold0/seed43 | 90.1697% | 90.0655% | -0.1042%p | 97.0060% / 83.1250% | [[399, 81], [5, 162]] |
| fold1/seed42 | 88.7566% | 90.6782% | +1.9215%p | 85.1064% / 96.2500% | [[462, 18], [28, 160]] |

고유 검증12환자·1315장, cached CPU 분석0.853초다. 기존 checkpoint의 최적 epoch 자체가 전체 validation으로 선택돼 이 LOPO는 그 선행 의존성을 없애지 못한다. **독립 Test·편향 없는 일반화 평가·nested 검증이 아니다.** 같은 fold0 두 seed를 다른 환자 집단으로 세지 않는다. 같은 전체 validation의 oracle 최적값을 주 결과로 사용하지 않았으며 grid 밖 tail/finer threshold는 시험하지 않았다. 환자에 따라 하락하고 클래스 편향·양성 미검출도 남는다. 위 fixed-.5 학습 변경들과 직접 우열 순위를 매기지 않는다.

## 코드 버전·SHA·모델 배포 범위

버전 코드는 [task2/experiments/20261125](../../task2/experiments/20261125/README.md)에 분리했다. 원래 학습에서 고정한13개 source의 hash를 보존하되 로컬 환자 감사 스크립트3개는 공개하지 않는다. 일반 학습/추론 소스10개는 해당 snapshot을 사용하고 팀 경로 config와 원래 config의 차이는 명시한다. 주요 원래 학습 source SHA256은 아래와 같고, 나머지는 실행별 `source_hashes.json`을 따른다.

- `common.py`: `649001dd765208000f541f7ce61670d7e503a99db34fa98fa01b7cb0ce4a7f41`
- `data.py`: `7cf24fab31da984f64def13fd2c43ef2e3b9afc06f2906cb660ccdfb8136273a`

| run ID | 원본 checkpoint SHA256 앞12자리 | 고정 원본 config SHA256 앞12자리 |
|---|---|---|
| resnet18_20261125_rotation_f0_s42 | 92efa4ccdb94 | cb811b51530f |
| resnet18_20261125_lr1e4_f0_s43 | 594515962ceb | cde2452219be |
| resnet18_20261125_brightness_f0_s42 | 6b73686133bf | 58b4bc2c7e3d |
| resnet18_20261125_contrast_f0_s42 | 9f25aff5ac3c | 5205a059401b |
| resnet18_20261125_rotation_f0_s43 | 44214e8dd0ad | f52cee35cd01 |
| resnet18_20261125_rotation_f1_s42 | 0548f2304b52 | a148ab2265af |
| resnet18_20261125_brightness_f1_s42 | 0c5be6e333d9 | 12e1d7e8b292 |
| resnet18_20261125_brightness_f0_s43 | 695027b11c71 | 5d1cc30bc671 |

표의 checkpoint hash는 **원본 로컬 모델**의 hash다. 새 공개 복사본은 개인 절대 경로가 든 `data_root`/`output_root` 두 값만 portable하게 바꾸므로 bytes와 공개 SHA가 다르다. 원본 로컬 모델은 보존하며 원본/공개 checkpoint SHA·tensor 및 추론 동등성·ZIP 체크섬/실제 패키지 검사 범위는 [모델 manifest](../../models/task2-augmentation-lr-20261125/manifest.json)와 해당 모델 검증 기록을 따른다. 기존20개 모델은 변경하지 않는다. 새 모델을 “원본 bytes 그대로”라고 주장하지 않는다.

run별 `history.csv`, `best_metrics.json`, 경로를 정리한 `config_recorded.json`, 재실행용 `config.json`, 환경의 패키지 버전·학습 요약·source hash·집계 validation checks를 제공한다. 원본 영상·라벨·patient/sample 목록·개별 예측·split·로컬 절대 경로·비밀정보·optimizer/scheduler/RNG state·영상 포함 노트북은 공유하지 않는다. best 모델은 추론용이며 로컬 last state와 구분한다. last state의 CPU 구조 검사는 이번 실제 run을 중단·재개한 시험을 뜻하지 않는다.

## 팀 재실행 예시

수업에서 허가받은 데이터는 저장소 `data/task2/`에 각자 준비한다. 기존 `task2/code`를 덮어쓰지 않고 별도 버전 코드를 사용한다. **이 버전 config의 기준은 `task2/experiments/20261125/`**이며 `data_root=../../../data/task2`, `output_root=../../../runs/task2/20261125`다. 저장소 루트에서:

```bash
python task2/experiments/20261125/code/resnet.py --config task2/experiments/20261125/configs/rotation_f0_s42.json --device cuda --run-name your_unique_rotation_run
python models/task2-augmentation-lr-20261125/download.py --list
# 새 Release가 실제 공개된 후
python models/task2-augmentation-lr-20261125/download.py --run resnet18_20261125_lr1e4_f0_s43 --output downloads/task2/lr1e4-s43
python downloads/task2/lr1e4-s43/predict.py --images input_images --output predictions.csv --device cpu
```

다른 후보는 해당 version config와 새 run 이름을 사용한다. 원본과 같은 코드·config·환경에서 직접 만든 별도 last state가 있을 때만 epoch 경계 resume를 사용한다. 공유 best 가중치만으로 optimizer/scheduler/epoch를 복원하지 않는다. GPU bitwise 재현을 보장하지 않는다.

## 자료 검색의 실제 확인 범위

이번 공유 작업은 기존 검토 기록을 다시 확인한 것이며 새 웹 검색이나 논문 성능 재현이 아니다. [기존 자료 검토](../task2-resnet18-20261111/RESEARCH_AND_NEXT_STEPS.md)에 링크·채택/제외 이유가 있다.

- [Principled Ultrasound Data Augmentation](https://arxiv.org/abs/2103.07895): 기존 기록에서 전문 방법/실험을 열람했으나 저자 공식 구현은 찾지 못했다. Task2에 같은 개선을 보증하지 않는다.
- [BackMix](https://papers.miccai.org/miccai-2024/092-Paper2248.html): 기존 전문 검토는 있으나 당시 repo 실제 소스 접근은 실패했다. mask/라벨·문제 차이로 도입하지 않았다.
- [UltraAugment](https://github.com/FlorianRamakers123/UltraAugment): 공식 코드 일부만 열람했고 본문/부록 접근은 실패했다. 전문·전체 재현 코드 검토 완료로 표시하지 않으며 fan geometry 전제가 맞는지 불명확해 도입하지 않았다.

기본 기하 정책·이번 밝기/대비는 Task2에 대한 자체 비교이며 논문의 데이터·설정에서 결과를 재현한 것이 아니다. 논문은 과제 규칙 허가를 대체하지 않는다.
