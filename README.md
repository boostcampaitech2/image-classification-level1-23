# Mask Image Classification

## 프로젝트 개요

- 더 높은점수를 내기 위해 여러가지 많은 검색들과 실험을 통해 base가 되는 Code 작성과, 많은 논문을 보는 기회가 될거라고 생각됩니다.
- 딥러닝의 베이스가 되는 Image Classification Competition을 통해  pytorch와 python를 사용하여 End-to-End로 코딩을 진행합니다. Competition이 끝난 후에 팀원 전부가 향상된 자신의 실력을 기대 할 수 있습니다.
- 이론으로 배웠던 내용을 파이토치 라이브러리를 사용하여 구현하는 능력을 기를 수 있을거라 기대합니다. 

<br>

## 프로젝트 수행 절차 및 방법

1주차 : 개인적으로 여러가지 Test를 진행함으로써 아키텍처를 선택  
2주차(월, 화): 아키텍처를 선택한 후 Image Augmentation에 대해서 토의 및 결과 공유 및 StartifiedFold 구현 공유   
2주차(수, 목): 성능향상을 위해 여러가지 기법(TTA, OOF,  Ensemble) Test 진행 및 결과 제출  
2주차(금,토): 리포트 작성  

<br>

## 프로젝트 수행 결과

1)  제출 결과 : 11위 (F1: 0.757, Accuracy: 79.476)
2)  학습 데이터
- 2700개의 폴더에 마스크 착용(5개), 미착용(1개), 정확하지 않은 착용(1개로) 총 18900개의 이미지
- 나이(3), 마스크 유무(3), 성별(2)에 따라 총 18개의 클래스

3) 모델 선정 : EfficientNet_b4 

VGG19 / ResNet 18 / ResNet50 / ResNext101 / ViT_224x224 / ViT 384 x 384 / EfficientNetB0 / EfficientNetB3 / EfficientNetB4 를 사용하여 모델을 구축했었고, timm 라이브러리의 tf_efficientnet_b4가 가장 좋은 성능을 보여주었습니다. 

<br>

## 아키텍쳐

* 아키텍쳐 : Efficientnet b4  
* Accuracy :79.476%  / F1 score : 0.757  
* Augmentation : Albumentation
  - Train_set  
CenterCrop(400,200), RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5), Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), p=1.0), ToTensorV2(p=1.0)
  - Validation_set : CenterCrop(height = 400, width = 200, Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), p=1.0), ToTensorV2(p=1.0)
* Loss
  - FocalLoss : EDA를 해본 결과 0.8% 비율인 클래스가 있고, 20%인 클래스도 있었습니다. 이런 불균형한 데이터 분포 데이터셋을 효과적으로 처리하기위해 FocalLoss를 직접 정의하여 사용하였습니다.
* Optimizer : AdamP
  - Learning rate  = 3e-4 적용
  - Weight decay = 1e-5 적용
*Scheduler
  - MultiStepLR (gamma =0.5)
* Datalode
  - train_loader( Batch_size = 32, Shuffle = True, Num_worker = 0, Drop_last = True)
  - validation_loader(Batch_size = 32, Shuffle = True  -> 해당 부분이 성능에 있어서 영향을 주었다는 사실이 새로 깨달은 부분이었다. Num_worker = 0)
  - test_loader(Batch_size = 32, Shuffle = False, Num_worker = 0)

<br>

## 검증 전략
Stratified 5-fold를 적용

<br>

## 앙상블 방법
Out-of-Fold : Evaluation단에서 최대한의 성능을 끌어내기 위해 같은 모델의 체크포인트를 앙상블하여 평균을 내는 방법을 사용하였습니다. 그 결과 f1_score 기준으로 0.1정도 향상되었습니다.

<br>

## 추가 시도
* 기존 age 라벨링 기준을 변경하였습니다. (60세이상 -> 58이상)
* Loss function을 CrossEntropy -> Focal Loss로 변경하였습니다.
* Stratified 5-fold를 적용해 F1 score :0.736 -> 0.750, Accuracy : 73.238->78.873로 성능을 높였습니다.
* Out-of-Fold 앙상블로 Evaluation단에서 최대한의 성능을 끌어내기 위해 같은 모델의 체크포인트를 앙상블하여 평균을 내는 방법을 사용하였습니다. 그 결과 F1 score 0.754, Accuracy 79.175로 성능이 향상되었습니다.
* Multi-label classification : Modeling시 마지막 layer에서 age / gender / mask 각각 결과값을 도출였고, Stratified 5-fold시 18개의 class가 아닌, age label에 대해 index를 split하였고 그 결과  F1 score 0.757, Accuracy 79.476으로 성능을 향상시켰습니다.

<br>

## 자체 평가 의견 - 잘한점들, 시도 했으나 잘 되지 않았던 것들
* 잘한점들
  - K-fold를 적용하면, Overfiting문제가 발생하는데 이 부분을 해결하기위해 팀원들이 많은 노력을 하였습니다.(ex. 앙상블, early stopping, check point를 이용한 즉각적 중지)
  - 팀원들이 다 함께 모델을 한 단계씩 발전 시켜 나갔습니다. 
  - 각자 모델을 학습 시켜보며 학습이 잘 된 점, 잘 되지 않은 점을 매일 공유했습니다.
  - 서로 제출을 미루거나 독식하려 하지않고, 양보하며 공평하게 제출하였습니다.

* 시도 했으나 잘 되지 않았던 것들 / 아쉬웠던점
  - MTCNN와 Retinaface를 이용해서 얼굴만 추출했는데 오히려 그게 성능이 떨어졌다.
  - 학습데이터가 작다고 판단하여 pre-trained model의 일부 파라미터를 freezing시켜봤는데, 성능 차이가 나지 않았습니다.
  - age에 대한 validation 정확도가 낮아서, Mix-up과 label smoothing을 시도해봤는데,
성능차이가 별로 나지 않았습니다.
  - age에 대한 분류를 조금 더 잘 하기 위해 이마 부분 crop, 코~턱 부분 crop, 턱 ~ 목 부분 crop을 해서 각각 학습시킨 모델을 앙상블 해봤는데, 오히려 성능이 조금 떨어졌습니다.
  - 현재 SOTA 모델로 알려진 ViT의 경우 Fixed Input을 받는 모델이었는데, ViT는 EfficientNet의 성능을 넘은 적이 없었습니다.
  - 60세 이상은 외부 데이터를 합쳐서 모델 성능을 개선시키려고 했으나, MaskTheFace 마스크 데이터셋을 제작했지만 incorrect 라벨이 부족해서 성능 개선에 한계가 있던 것 같습니다.
  - Ipynb파일로 실험을 진행했는데 py파일로 실험하지 못한게 아쉬웠습니다.

## 사용 기술 및 도구
- VsCode
- JupyterNotebook
- python (3.8)
- conda (4.10.3)
- pytorch (1.7.1)
- pandas  (1.3.2)
- numpy (1.19.2)

