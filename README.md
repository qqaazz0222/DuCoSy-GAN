# DUCOSY-GAN: Dual HU-Range Complementary Synthesis GAN

High-Fidelity Contrast-Enhanced CT Generation Using Complementary Synthesis of Dual HU-Range Targeted GANs

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
├── checkpoints
│   ├── Lung_Generator_A2B.pth
│   ├── Lung_Generator_B2A.pth
│   ├── Soft_Tissue_Generator_A2B.pth
│   └── Soft_Tissue_Generator_B2A.pth
├── data
│   ├── input
│   ├── output
│   └── working
├── modules
│   ├── __init__.py
│   ├── argmanager.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocess.py
│   └── trainer.py
├── training_dir
├── README_EN.md
├── README.md
├── requirements.txt
├── train.py
├── inference.py
└── anonymize.py
```

## Data

모델 학습 및 CT 영상 생성을 위해 아래와 같이 데이터 디렉토리를 구성합니다.

```bash
.
└── data
    └── input
        └── {Dataset Name}
            ├── {Patient ID}
            │   ├── POST STD # 학습시에만 필요
            │   │   ├── {slice_0}.dcm
            │   │   └── ...
            │   └── POST VUE
            │       ├── {slice_0}.dcm
            │       └── ...
            └── ...
```

## Train

모델 학습을 위해 아래 명령어를 실행합니다.

```bash
python train.py
```

## Generate

CT 영상 생성을 위해 아래 명령어를 실행합니다.

```bash
python generate.py
```

## Anonymize

생성된 CT 영상에 익명화를 적용하기 위해 아래 명령어를 실행합니다.

```bash
python anonymize.py
```
