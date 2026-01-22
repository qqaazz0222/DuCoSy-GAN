# DuCoSy-GAN: Dual HU-Range Complementary Synthesis GAN

High-Fidelity Contrast-Enhanced CT Generation Using Complementary Synthesis of Dual HU-Range Targeted GANs

## Overview

DuCoSy-GAN은 비조영 CT(NCCT)로부터 고품질의 조영 증강 CT(CECT)를 생성하기 위한 딥러닝 모델입니다. 본 연구는 서로 다른 두 개의 HU(Hounsfield Unit) 범위에 특화된 CycleGAN 모델을 학습하고, 이를 상보적으로 합성하여 전체 HU 범위에 걸친 고품질 CECT 영상을 생성합니다.

### 주요 특징

- **Dual HU-Range Targeting**: Soft-tissue(-150~250 HU)와 Lung(-1000~-150 HU) 영역을 각각 독립적으로 학습
- **Complementary Synthesis**: 두 모델의 출력을 상보적으로 합성하여 전체 HU 범위 커버
- **CBAM Attention Mechanism**: 채널 및 공간 어텐션을 활용한 특징 강화
- **Anatomical Mask Integration**: 뼈 및 종격동 마스크를 입력에 통합하여 해부학적 정보 활용
- **Contrast-Focused Loss Functions**: 조영 효과에 집중한 맞춤형 손실 함수

### Note

모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 25.5.1

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success) 또는 [🔗 미니콘다 다운로드](https://www.anaconda.com/docs/getting-started/miniconda/main)

**Step 1**. 저장소 복제

```bash
git clone https://github.com/qqaazz0222/DUCOSY-GAN
cd DUCOSY-GAN
```

**Step 2**. Conda 가상환경 생성 및 활성화

```bash
conda create --name ducosygan python=3.12 -y
conda activate ducosygan
```

**Step 3**. 라이브러리 설치

```bash
pip install -r requirements.txt
```

## Structure

프로젝트 구조는 아래와 같습니다.

```bash
.
├── checkpoints                    # 사전 학습된 모델 가중치
│   ├── Lung_Generator_A2B.pth
│   ├── Lung_Generator_B2A.pth
│   ├── Soft_Tissue_Generator_A2B.pth
│   └── Soft_Tissue_Generator_B2A.pth
├── data                          # 데이터 디렉토리
│   ├── input                     # 입력 DICOM 파일
│   ├── output                    # 최종 생성 결과
│   └── working                   # 중간 처리 파일
├── modules                       # 핵심 모듈
│   ├── __init__.py
│   ├── argmanager.py            # 명령행 인자 관리
│   ├── dataset.py               # 데이터셋 로더 및 전처리
│   ├── model.py                 # Generator/Discriminator 정의
│   ├── preprocess.py            # DICOM 전처리 유틸리티
│   ├── mask_generator.py        # 자동 마스크 생성
│   └── trainer.py               # 학습 루프 및 손실 함수
├── training_dir                  # 학습 중 생성되는 파일
│   ├── soft_tissue              # Soft-tissue 모델 학습
│   │   ├── images               # 검증 이미지
│   │   └── saved_models         # 체크포인트
│   └── lung                     # Lung 모델 학습
│       ├── images
│       └── saved_models
├── README.md
├── requirements.txt
├── train.py                     # 학습 스크립트
├── generate.py                  # 추론 및 영상 생성
├── masking.py                   # 심장 마스킹
└── anonymize.py                 # DICOM 익명화
```

## Data

모델 학습 및 CT 영상 생성을 위해 아래와 같이 데이터 디렉토리를 구성합니다.

```bash
.
└── data
    └── input
        └── {Dataset Name}              # 예: Kangwon_National_Univ_Chest
            ├── {Patient ID}            # 예: 00000102-1542-11-06
            │   ├── POST STD            # CECT (학습 시에만 필요)
            │   │   ├── {slice_0}.dcm
            │   │   ├── {slice_1}.dcm
            │   │   └── ...
            │   └── POST VUE            # NCCT (필수)
            │       ├── {slice_0}.dcm
            │       ├── {slice_1}.dcm
            │       └── ...
            ├── {Patient ID}
            │   └── ...
            └── ...
```

### 데이터 요구사항

- **학습(Train)**: POST VUE(NCCT)와 POST STD(CECT) 폴더 모두 필요
- **추론(Generate)**: POST VUE(NCCT) 폴더만 필요
- **파일 형식**: DICOM(.dcm)
- **정렬 방식**: InstanceNumber 또는 SliceLocation 메타데이터로 자동 정렬

## Model Architecture

### Generator (ResNet-based with CBAM)

DuCoSy-GAN의 Generator는 ResNet 기반 구조에 CBAM(Convolutional Block Attention Module)을 통합하여 구현되었습니다.

**주요 구성 요소:**

1. **Encoder (Down-sampling)**
   - 초기 Reflection Padding + 7×7 Convolution
   - 2단계 Down-sampling (stride=2)
   - 채널 수: 64 → 128 → 256

2. **Transformation (Residual Blocks)**
   - 9개의 Residual Blocks (256 channels)
   - 각 블록에 CBAM Attention 적용
   - Channel Attention: 채널별 특징 가중치 학습
   - Spatial Attention: 공간적 중요 영역 강조

3. **Decoder (Up-sampling)**
   - 2단계 Up-sampling (scale_factor=2)
   - 채널 수: 256 → 128 → 64
   - 최종 7×7 Convolution + Tanh 활성화

4. **Input/Output**
   - Input channels: 1 (CT image) + N (anatomical masks)
   - Output channels: 1 (generated CT image)
   - Image size: 512×512

### Discriminator (PatchGAN)

PatchGAN 구조를 사용하여 이미지의 국소적 진위를 판별합니다.

**구조:**
- 4단계 Down-sampling
- 채널 수: 64 → 128 → 256 → 512
- LeakyReLU 활성화 (negative slope=0.2)
- 출력: 32×32 패치별 진위 판별

### Mask Integration

해부학적 구조 정보를 활용하기 위해 자동 생성된 마스크를 입력에 통합합니다.

**지원되는 마스크 타입:**
- **Bone Mask**: 뼈 영역 (HU > 200)
- **Mediastinum Mask**: 종격동 영역 (특정 공간 범위)
- 자동 생성 또는 사전 저장된 DICOM 마스크 사용 가능

## Train

모델 학습을 위해 아래 명령어를 실행합니다.

```bash
python train.py
```

### 학습 프로세스

DuCoSy-GAN은 두 개의 독립적인 CycleGAN 모델을 학습합니다:

1. **Soft-tissue Model** (HU: -150 ~ 250)
   - 연조직, 혈관, 조영 증강 영역에 집중
   - 조영제 효과가 가장 두드러지는 영역

2. **Lung Model** (HU: -1000 ~ -150)
   - 폐 실질 및 기도에 집중
   - 저밀도 영역의 세밀한 구조 보존

### 학습 설정

**기본 하이퍼파라미터:**
```python
epochs = 10000              # 총 에포크 수
decay_epoch = 100           # 학습률 감소 시작 에포크
batch_size = 8              # 배치 크기
lr = 0.0002                 # 초기 학습률
lambda_cyc = 10.0           # Cycle consistency loss 가중치
lambda_id = 5.0             # Identity loss 가중치
img_size = 512              # 입력 이미지 크기
val_split = 0.2             # 검증 데이터 비율
```

**손실 함수:**
- **GAN Loss**: Adversarial loss (MSE)
- **Cycle Consistency Loss**: Forward-backward 재구성 손실
- **Identity Loss**: 동일 도메인 입력에 대한 불변성
- **Gradient Loss**: 경계 선명도 유지
- **SSIM Loss**: 구조적 유사성
- **Contrast-focused Losses**:
  - Contrast Attention Loss: 조영 차이 영역 집중
  - Contrast Region Loss: 조영 증강 영역 재현
  - Contrast Edge Loss: 조영 경계 선명도

### 학습 모니터링

학습 중 다음 정보가 생성됩니다:

1. **체크포인트** (`training_dir/{model}/saved_models/`)
   - `checkpoint.pth.tar`: 최신 모델 상태
   - `best_G_A2B.pth`, `best_G_B2A.pth`: 최적 Generator 가중치
   - Optimizer 상태 및 학습 이력 포함

2. **검증 이미지** (`training_dir/{model}/images/`)
   - 에포크별 생성 결과 이미지
   - 형식: `epoch_{N}.jpg`
   - 배치: [NCCT | Generated CECT | Real CECT]

3. **로그 파일**
   - 학습 손실 및 검증 손실 기록
   - GPU 메모리 사용량 모니터링

### 다중 GPU 학습

8개의 GPU를 사용하는 DataParallel 학습이 자동으로 활성화됩니다:

```python
# 자동 감지 및 활성화
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
G_A2B = nn.DataParallel(G_A2B, device_ids=gpu_ids)
```

### 학습 재개

중단된 학습을 재개하려면 `resume` 옵션을 사용합니다:

```bash
python train.py --resume checkpoint.pth.tar
```

## Generate

CT 영상 생성을 위해 아래 명령어를 실행합니다.

```bash
python generate.py
```

### 생성 프로세스

이 과정에서 NCCT 영상을 로드하여 아래 단계를 수행합니다.

#### 1. 각 HU Range별 영상 생성 (`generate`)

**Soft-Tissue Model:**
- 입력: NCCT (HU: -150 ~ 250 범위로 정규화)
- 출력: Soft-tissue CECT 생성 결과
- 저장 위치: `data/working/{Dataset}/{PatientID}/soft_tissue/`

**Lung Model:**
- 입력: NCCT (HU: -1000 ~ -150 범위로 정규화)
- 출력: Lung CECT 생성 결과
- 저장 위치: `data/working/{Dataset}/{PatientID}/lung/`

#### 2. 생성 영상을 상보적 합성 (`integrate`)

합성 과정은 다음과 같이 진행됩니다:

**a. 데이터 로드 및 변환**
```python
# Raw Pixel Array를 HU Pixel Array로 변환
hu_array = raw_array * RescaleSlope + RescaleIntercept
```

**b. HU 범위별 합성**
- **Lung Range** (-1000 ~ -150 HU): Lung 모델 출력 사용
- **Soft-tissue Range** (-150 ~ 250 HU): Soft-tissue 모델 출력 사용
- **Out-of-Range** (< -1000, > 250 HU): 원본 NCCT 값 유지

**c. DICOM 메타데이터 보존**
- 원본 DICOM의 모든 메타데이터 유지
- Pixel Array만 생성된 CECT로 교체
- RescaleSlope, RescaleIntercept 재계산

**d. 최종 저장**
- 저장 위치: `data/output/{Dataset}/{PatientID}/POST_STD/`
- 형식: DICOM (.dcm)

### 생성 옵션

**사용 가능한 주요 옵션:**

```bash
# 특정 데이터셋만 처리
python generate.py --dataset_names "Dataset1,Dataset2"

# GPU 선택
python generate.py --gpu_id 0

# 커스텀 모델 경로
python generate.py \
  --soft_tissue_model_path "path/to/soft_tissue.pth" \
  --lung_model_path "path/to/lung.pth"

# 이미지 크기 조정
python generate.py --img_size 512
```

### 후처리 옵션

생성된 영상의 품질 향상을 위한 후처리 옵션:

1. **Difference Map 적용** (`apply_diffmap`)
   - 원본과 생성 영상의 차이를 분석하여 조영 효과 강조
   - 노이즈 감소 및 조영 일관성 향상

2. **Volume Post-processing**
   - 슬라이스 간 연속성 보정
   - 3D 일관성 유지

### 출력 구조

```bash
data/
├── working/                      # 중간 결과물
│   └── {Dataset}/
│       └── {PatientID}/
│           ├── raw/             # 원본 NCCT 복사본
│           ├── soft_tissue/     # Soft-tissue 생성 결과
│           └── lung/            # Lung 생성 결과
└── output/                      # 최종 결과물
    └── {Dataset}/
        └── {PatientID}/
            └── POST_STD/        # 합성된 CECT
                ├── {slice_0}.dcm
                ├── {slice_1}.dcm
                └── ...
```

## Masking

CT 영상에서 본 연구의 범위가 아닌 심장 및 심혈관계를 마스킹하기 위해 아래 명령어를 실행합니다.

```bash
python masking.py
```

### 마스킹 옵션

동시 처리 수를 `--batch_size=n`으로 수정할 수 있습니다.

```bash
# 배치 크기 조정 (기본값: 4)
python masking.py --batch_size 4

# 특정 데이터셋만 처리
python masking.py --dataset_names "Dataset1,Dataset2"
```

**성능 참고사항:**
- `batch_size=4`일 경우 약 22GB의 GPU 메모리 사용
- RTX 4090 기준 환자당 50~80초 소요
- 메모리 부족 시 배치 크기를 줄이세요 (예: `--batch_size 2`)

### 마스킹 프로세스

이 과정에서 [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)를 활용하여 심장 및 심혈관 마스킹을 생성합니다.

#### 1. 마스킹 생성 (`generate`)

**자동 세그멘테이션:**
- TotalSegmentator를 사용하여 104개 해부학적 구조 세그멘테이션
- 심장 관련 구조 추출:
  - Heart (심장)
  - Aorta (대동맥)
  - Pulmonary artery (폐동맥)
  - Pulmonary vein (폐정맥)
  - Superior/Inferior vena cava (상/하대정맥)

**마스크 저장:**
- 위치: 각 환자 디렉토리 내 `heart_mask/` 폴더
- 형식: DICOM (.dcm), 이진 마스크 (0 또는 1)

#### 2. 마스킹 적용 (`apply`)

**마스크 적용 방법:**
```python
# 마스크 영역을 원본 NCCT 값으로 대체
masked_cect[mask == 1] = original_ncct[mask == 1]
```

이를 통해:
- 심장/혈관 영역에서는 조영 효과 제거
- 나머지 영역은 생성된 CECT 유지
- 연구 대상인 폐 및 흉부 연조직에만 집중

### 마스킹 워크플로우

```bash
# Step 1: 원본 데이터 준비
data/input/{Dataset}/{PatientID}/POST_VUE/

# Step 2: CECT 생성
python generate.py

# Step 3: 마스크 생성 및 적용
python masking.py

# Step 4: 마스킹된 CECT 출력
data/output/{Dataset}/{PatientID}/POST_STD_MASKED/
```

### TotalSegmentator 설치

마스킹 기능을 사용하려면 TotalSegmentator가 설치되어 있어야 합니다:

```bash
pip install TotalSegmentator
```

**시스템 요구사항:**
- CUDA 지원 GPU (최소 8GB VRAM 권장)
- PyTorch 1.7 이상
- 충분한 디스크 공간 (모델 가중치 약 300MB)

## Anonymize

생성된 CT 영상에 익명화를 적용하기 위해 아래 명령어를 실행합니다.

```bash
python anonymize.py
```

### 익명화 프로세스

이 스크립트는 DICOM 파일의 개인정보 메타데이터를 제거하거나 익명화합니다.

**익명화되는 필드:**
- Patient Name
- Patient ID
- Patient Birth Date
- Patient Sex
- Patient Age
- Study Date/Time
- Series Date/Time
- Acquisition Date/Time
- Institution Name
- Referring Physician Name
- 기타 식별 가능한 정보

**보존되는 필드:**
- 영상 관련 메타데이터 (Pixel Spacing, Slice Thickness 등)
- 스캔 파라미터 (kVp, mAs 등)
- HU 변환 정보 (Rescale Slope/Intercept)

### 익명화 옵션

```bash
# 특정 데이터셋만 익명화
python anonymize.py --dataset_names "Dataset1,Dataset2"

# 출력 디렉토리 지정
python anonymize.py --output_dir "data/anonymized"

# 익명 ID 패턴 설정
python anonymize.py --patient_id_prefix "ANON"
```

### 사용 시나리오

1. **연구 데이터 공유**
   - 생성된 CECT를 다른 기관과 공유할 때
   - 개인정보 보호 규정(HIPAA, GDPR) 준수

2. **공개 데이터셋 구축**
   - 벤치마크 데이터셋 생성
   - 오픈소스 의료 영상 데이터베이스

3. **모델 평가**
   - 외부 검증용 익명화된 데이터셋

## Performance & Results

### 학습 시간

**단일 모델 학습 (RTX 4090 × 8):**
- Soft-tissue Model: 약 48시간 (10,000 epochs)
- Lung Model: 약 48시간 (10,000 epochs)

**데이터셋 크기별:**
- 환자 55명, 슬라이스 약 4,849개 (학습) + 1,229개 (검증)
- 배치 크기 8, epoch당 약 33분

### 추론 시간

**영상 생성 속도 (RTX 4090 × 1):**
- 환자당 평균 처리 시간: 5~10초
- 슬라이스당 평균 처리 시간: 0.1~0.2초
- 배치 추론으로 더 빠른 처리 가능

### 메모리 요구사항

**학습:**
- GPU: 약 20GB per GPU (8 GPUs with DataParallel)
- System RAM: 32GB 이상 권장

**추론:**
- GPU: 약 8GB
- System RAM: 16GB 이상 권장

## Troubleshooting

### CUDA Out of Memory

**해결 방법:**
```bash
# 배치 크기 감소
python train.py --batch_size 4

# 이미지 크기 감소
python train.py --img_size 256

# 워커 수 감소
python train.py --num_workers 8
```

### DataParallel 관련 오류

**모듈 이름 불일치:**
```python
# 체크포인트 로드 시 'module.' 접두사 제거
if all(key.startswith('module.') for key in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
```

### 마스크 생성 오류

**TotalSegmentator 설치 확인:**
```bash
pip install --upgrade TotalSegmentator
totalseg --version
```

**GPU 메모리 부족:**
```bash
# 배치 크기 감소
python masking.py --batch_size 2
```

## Citation

이 연구를 사용하시는 경우 아래와 같이 인용해 주세요:

```bibtex
@article{ducosygan2026,
  title={DuCoSy-GAN: High-Fidelity Contrast-Enhanced CT Generation Using Complementary Synthesis of Dual HU-Range Targeted GANs},
  author={Author Names},
  journal={Journal Name},
  year={2026}
}
```

## License

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 라이선스를 따릅니다.

## Acknowledgments

- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for the base architecture
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for anatomical segmentation
- [PyTorch](https://pytorch.org/) for deep learning framework
