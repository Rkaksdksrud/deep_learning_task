# 지금까지의 자료 검토와 다음 시도

정리일: 2026-11-13. 11/11의 검색·실험 기록과 11/13의 후속 계획을 요약했다. 새 검색·새 학습·추가 holdout 평가는 하지 않았다. 합성입력 기반 배포검사는 [package_validation.json](../../models/task2-resnet18-20261111/package_validation.json)을 참고한다. 링크는 당시 확인한 자료이며, **아래 후속 실험은 모두 계획·미실행**이다.

## 완료한 결과와 해석

ResNet18 전체 7개 run의 학습·독립검증을 완료했다(95 epoch·학습 2,308.8초). Train 수평반전 p=0.5 + ±7° 회전 정책은 세 matched 조건에서 BA 평균 57.57→89.80%(+32.23%p)로 개선돼 채택했다. 일반 CE는 한 조건 BA 59.65%(+12.10%p)만 확인해 **미채택·안정성 미확인**이다. 상세 수치는 [실험 README](README.md)와 [집계](metrics_summary.json)에 있다.

개선은 반전·회전 및 보간/크기 변화가 포함된 **정책 전체 효과**다. 개별 연산 기여나 논문 성능 재현은 아니다. 같은 fold0 환자를 두 seed에서 반복한 세 조건이며 독립 3-fold나 private Test 결과가 아니다. fold0 실패 환자 BA 50%, fold1 실패 환자 BA 26.25%가 남았다. VGG·GoogLeNet의 전체 학습과 전체 데이터 최종 모델은 미완료다.

## 이미 확인한 핵심 자료

기존 모델의 훈련/환자 holdout 격차와 환자 간 순위 역전 이후 검색했다. 직접적인 혈관 내 sheath Present/Absent와 일치하면서 유력한 연구는 당시 찾지 못했다. 다른 초음파 분류·분할 결과를 같은 과제의 증거로 취급하지 않는다.

| 자료·링크 | 당시 실제 접근 범위 | 적용 판단 |
|---|---|---|
| **Principled Ultrasound Data Augmentation** — [공식 페이지](https://arxiv.org/abs/2103.07895), [전문](https://arxiv.org/pdf/2103.07895) | 전문 12쪽의 방법·실험·표·한계를 텍스트로 읽었다. 표 이미지 시각 확인은 실패했고 저자 공식 구현은 찾지 못했다. | 태아 표준단면 14분류다. 환자 분리·외부검증·반복 seed 근거가 불충분하고 증강 요소가 얽혀 있다. 간단한 Train 증강 비교의 간접 근거로만 활용했다. MixRA/mixing·대규모 정책 탐색은 라벨 의미·규칙·비용 때문에 미도입. |
| **BackMix, MICCAI 2024** — [공식 페이지](https://papers.miccai.org/miccai-2024/092-Paper2248.html), [전문](https://papers.miccai.org/miccai-2024/paper/2248_paper.pdf), [저자 GitHub](https://github.com/kitbransby/BackMix) | 전문과 repo README·파일 목록을 확인했다. 학습/utils/models 소스는 접근 실패해 코드 일치 검증은 못 했다. | 심장 4-view 분류이며 환자 분리·3 seed·외부평가가 있다. 하지만 baseline에도 표준 증강이 있어 flip/회전 효과를 분리하지 않는다. 필요한 sector mask·추가 라벨 허용과 실패원인 근거가 부족해 핵심 배경교환 방법은 미채택. |
| **UltraAugment, CVPRW 2024** — [공식 페이지](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Ramakers_UltraAugment_Fan-shape_and_Artifact-based_Data_Augmentation_for_2D_Ultrasound_Images_CVPRW_2024_paper.html), [저자 GitHub](https://github.com/FlorianRamakers123/UltraAugment) | 초록·README와 증강/polar 변환/dataloader/setup 일부 소스를 읽었다. 본문·부록·실제 학습 config/fold 내용은 접근 실패했다. **전문 검토 완료가 아니다.** | fan 좌표·마스크 전제가 현재 영상에서 확인되지 않았다. letterbox 패딩은 acquisition fan mask가 아니다. 코드의 환자 키 기본 동작만으로 실제 분리도 확정할 수 없다. 추가 왜곡·의존성·검증 비용 대비 근거 부족으로 직접 도입 기각. |
| **HF timm ResNet18** — [모델 카드](https://huggingface.co/timm/resnet18.a1_in1k), [공식 ResNet 코드](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/resnet.py) | 카드와 BasicBlock/ResNet 정의를 읽었다. | ImageNet-1k/A1 레시피 참고만 했다. 외부 가중치 미사용이며 모든 timm 설정이 현재 TorchVision과 같다고 주장하지 않는다. |
| **HF 사용자 fetal-plane ResNet18** — [모델 카드](https://huggingface.co/Marc-HealthAI/Crimson-Flame-6-resNet18) | 카드·파일 목록만 확인했다. 학습 소스는 찾지 못했고 config 접근도 실패했다. | 연구 공식 구현으로 확인되지 않았고 환자 분리·seed·초기화 정보가 부족해 점수 비교·가중치 채택 제외. |

지정 구조/API는 기존에 확인한 TorchVision v0.25.0의 [VGG](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/vgg.py), [ResNet](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/resnet.py), [GoogLeNet](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/googlenet.py)을 기준으로 삼았다. GoogLeNet은 설치 소스도 확인했고 [원 논문 §5](https://arxiv.org/html/1409.4842v1)의 보조 손실 계수 0.3을 참고했다. 일반 CE는 [공식 손실 설명](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)과 코드 호출을 확인한 진단 비교다. 조사한 추가 연구 repo의 코드·데이터·가중치는 도입·실행하지 않았다.

## 다음 우선순위와 판정 — 계획·미실행

1. **GoogLeNet → VGG 전체 학습·독립검증.** 첫 예산은 모델별 1개, 합 2개 run이다. random init, fold0/seed42, RGB224, weighted CE, 현재 기하 정책, threshold 0.5, 최대30epoch/patience7로 시작한다. GPU/VRAM·몇 batch 동작을 확인해 본 학습 전에 batch를 고정한다. 다른 모델에서 증강 효과를 주장하려면 그 모델의 무증강 대조군도 필요하다.
2. **고정 ResNet 실패 진단(새 학습 0회).** 무증강 Train-eval/Valid의 확률·logit·AUROC·혼동행렬을 비교한다. 필요시 BN 통계를 관측하되 상태를 바꾸거나 검증 환자로 통계를 갱신하지 않는다. 진단 근거가 생긴 변경만 새 후보로 올린다.
3. **반전만/회전만 비교.** 먼저 각도0·반전 없음 경로와 무증강 입력의 텐서 일치를 확인한다. 같으면 resize-only 학습은 생략하고 다르면 공통 대조가 필요하다. fold0/seed42 후보 2개를 비교하고 유망한 하나만 추가 fold/seed로 확인한다. 현재 정책보다 낫거나 비슷한 성능을 더 적은 비용으로 유지할 근거가 없으면 정책을 유지한다.
4. **224→320 후보 하나.** 기존 오프라인 재표본화 품질 차이는 학습 성능 증거가 아니다. 현재 정책에서 입력 크기만 바꿔 첫 1개 run을 비교한다. 같은 batch가 불가능하면 두 해상도를 같은 새 batch로 다시 비교하거나 보류한다. 후속 제안 기준은 두 개발 fold matched 평균 BA **+1%p 이상·두 조건 모두 무악화**, 두 recall·최악 환자·실측 비용 확인 후 다른 seed까지 검증하는 것이다. 비교 기준은 현재 기하 정책이며 예전 무증강 점수가 아니다.
5. **후보 고정 후 새 fold 제한 확인.** 목적을 Control/Geometry 안정성 또는 Geometry/신규 후보의 추가 효과 중 하나로 정해 paired 2개 run만 비교한다. 새 fold에서 최고 epoch를 고르면 추가 validation이다. 독립 확인은 inner group-valid에서 checkpoint를 정하거나 개발 결과로 epoch를 사전 고정한 뒤 outer를 1회 평가하는 설계가 필요하다.

위 실험을 한꺼번에 실행하지 않는다. **바로 다음 하나는 GoogLeNet 전체 학습과 저장 checkpoint 재검증**이다. 장시간 학습 전 optimizer/scheduler/RNG/current epoch 저장 보강을 검토한다. 현재 best checkpoint만으로 정확한 중간 이어학습은 불가능하다. 최종 레시피·학습 길이를 고정한 뒤 전체 데이터 모델과 정식 제출물을 준비한다.

## 공통 판단과 미확정 규칙

기존 정책의 사전 채택 기준은 세 matched 조건 평균 +2%p 이상·최소 두 조건 개선·각 조건 하락 2%p 이내였고 모두 충족했다. 이는 환자별 무악화 기준이 아니므로 실패 환자와 비용도 공개했다. 후속 변경은 현재 정책과 같은 분할·seed·예산으로 비교하고 BA·두 recall·환자별 악화·시간을 함께 본다. GPU는 한 작업씩, 작은 상승 한 번으로 채택하지 않으며 private Test로 반복 선택하지 않는다.

제공 데이터만 사용, 지정 모델 독립 학습, 회전/반전 허용, 앙상블·다른 신경망 특징추출/전처리 금지는 원문 점검에서 확인했다. **사전학습, 밝기/대비/노이즈·CLAHE·새 ROI, stem/정규화층 등 구조 변경, 추가 annotation·의사 라벨·mixing, threshold/TTA·BN 재보정, 외부 코드 활용의 과제 범위, 채점 loader 인터페이스는 미확정**이다. 공식 확인 전 해당 적용은 보류하고 random init·기본 구조·threshold 0.5를 유지한다.

일반 CE+기하 정책 결합, 더 깊은 모델, 자동 증강 탐색·복잡한 새 방법·무작정 epoch 증가는 현재 근거와 비용 대비 우선순위가 낮다. 환자 라벨 변경·실패 환자 제외·환자별 임계값도 적용하지 않는다.
