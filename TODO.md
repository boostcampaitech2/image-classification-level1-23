## Dataset & Dataloader

- [ ] Retinaface, dlib face detection -> else: centercrop: VERY IMPORTANT. Mask position on face determines incorrect vs correct
  - [ ] Mask detection model (mobilenet) -> face extraction
- [ ] Make face mask dataset
  - [ ] opencv mask / overlay mask on the face
	- [ ] 60살 이상의 노인 이미지를 인터넷에서 가져오는 게 필요할 것 같다.
  - [ ] [노인 이미지를 갖고 오고 Facial Landmark 기반으로 마스크 씌우기](https://www.youtube.com/watch?v=ncIyy1doSJ8)
	- [ ] Get mask & face dataset, move mask downwards, make it as incorrect dataset
- [ ] 아니면 Facial Landmark를 파란색 점으로 직접 이미지들에 찍을까?  그러면 AI 모델이 "신체 부위"로 학습을 할텐데.
	- [ ] 얼굴은 다 보이지 않는데 facial landmark를 일부분 찍을 방법이 있으려나?: (Mask가 있는데 입술만 보인다) or (Mask가 있는데 코가 보인다) -> incorrect
- [ ] Apply Different types of transformation to age/gender/mask
- [ ] add 59 years old, 58 years old to 60 years old and above class
	- [ ] Age Distribution(Age <= 20, Age <58, else)로 분류를 하는 게 좋을 것 같다.



## Training
- [ ] 연주님: 하나의 모델에서 레이어를 각각각 써서 하나의 결과값으로 나오게 하는 것. -> Multimodal
- [ ] Use SGD, SGDP as optimizer. SGD outperforms Adam.
- [ ] early stopping on age (epoch 10 is too much already)
- [ ] Try Multimodal of ViT on Age, EfficientNet on Mask and Gender


## Inference

- [ ] visualize the predicted result of the csv - label and picture
- [ ] Visualize using Confusion Matrix to check accuracy [example](https://github.com/snoop2head/ml_classification_tutorial/blob/main/ML_Classification.ipynb)

---

## Done

- [x] Make Inference Function
- [x] Fix Resnet Code
- [x] Apply Resnet Code
- [x] reallocate local dataset loader according to right path format(input/data not input/)
- [x] reallocate local dataset including jpeg, png format
- [x] Change the code, clean dataset on upstage server
- [x] check class order vs output prediction order -> fixed with class_to_idx
- [x] Make dev environment on colab
- [x] should i set shuffle=True for test data loader? -> No.
- [x] Start making from dataloader
  - [x] Getting F1 score
  - [x] Apply Augmentation (imgaug) -> Albumentation으로 대체
- [x] Apply ResNext50 example 1
- [x] Apply EfficientNet
- [x] Apply ViT
  - [x] https://github.com/lukemelas/PyTorch-Pretrained-ViT
  - [x] https://github.com/lucidrains/vit-pytorch
- [x] label 0 and 1 as classes, not integers. too time consuming to figure out which is which
- [x] Training set에 대해서 모든 Class에 대한 데이터 개수를 동일하게 설정하려고 했는데(Sampling) 그게 올바르지 않는 접근 방법이었다.
- [x] 9:1이랑 8:2의 dataset 나누기 차이가 없었다.
