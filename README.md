# pstage_01_image_classification

## Getting Started

Template for the competition.

### Team Links
- [📷 Zoom 회의실 23](https://zoom.us/j/97196865381?pwd=ckxjdkhLV3EzSEI5L3FhNC9WaVg3dz09)
- [📂 구글 드라이브(Drive)](https://drive.google.com/drive/u/2/folders/1oI71ZYms5crDxkE1xR9LryRzn45wTP4W)
- [🧪 성능 기록지](https://docs.google.com/spreadsheets/d/1dDS188VUSZ7-l7nRtegN4KoUHEO0o0lvpu855mYHUzY/edit?usp=sharing)
- [🎯 Google Group Admin Panel](https://groups.google.com/g/temp-boostcamp-ai/members)


### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
