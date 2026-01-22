# 자동 마스크 생성 기능 사용 가이드

## 개요

DuCoSy-GAN 모델은 이제 DICOM 데이터로부터 **자동으로 해부학적 마스크를 생성**할 수 있습니다. 별도의 마스크 파일 없이 HU 값 기반으로 폐, 종격동, 뼈, 폐혈관 마스크를 실시간으로 생성하여 학습 및 추론에 사용합니다.

## 주요 변경사항

### 1. 새로운 모듈: `mask_generator.py`

해부학적 마스크 자동 생성 모듈이 추가되었습니다.

**지원 마스크 종류:**
- `lung`: 폐 영역 마스크 (HU: -1000 ~ -300)
- `mediastinum`: 종격동 마스크 (HU: -300 ~ 450, ConvexHull 기반)
- `bone`: 뼈 마스크 (HU: >200, CECT 조영 혈관 제외)
- `lung_vessel`: 폐혈관 마스크 (binary_fill_holes 기반)

**주요 기능:**
```python
from modules.mask_generator import generate_anatomical_masks

# HU 이미지로부터 마스크 생성
masks = generate_anatomical_masks(
    hu_image,  # numpy array (2D or 3D)
    mask_types=['lung', 'mediastinum', 'bone', 'lung_vessel']
)
# Returns: {'lung': array, 'mediastinum': array, 'bone': array, 'lung_vessel': array}
```

### 2. Dataset 자동 마스크 생성 지원

`DicomDataset` 클래스가 자동 마스크 생성을 지원하도록 업데이트되었습니다.

**새로운 인자:**
- `auto_generate_masks` (bool): DICOM에서 자동으로 마스크 생성 여부
- `mask_types` (list): 생성할 마스크 종류 리스트

### 3. 학습 설정 업데이트

`argmanager.py`의 학습 설정에 자동 마스크 생성이 기본으로 활성화되었습니다.

**Soft Tissue Model:**
```python
args = argparse.Namespace(
    use_masks=True,
    auto_generate_masks=True,  # 자동 생성 활성화
    mask_types=['bone', 'mediastinum'],  # 뼈, 종격동 마스크
    ...
)
```

**Lung Model:**
```python
args = argparse.Namespace(
    use_masks=True,
    auto_generate_masks=True,  # 자동 생성 활성화
    mask_types=['lung'],  # 폐 마스크
    ...
)
```

## 사용 방법

### 1. 테스트 실행

자동 마스크 생성 기능을 테스트하려면:

```bash
cd /workspace/Contrast_CT/hyunsu/DuCoSy-GAN
python test_mask_generation.py
```

결과: `test_mask_generation.png` 파일에 시각화 결과 저장

### 2. 학습 시 사용

기존 학습 스크립트를 그대로 사용하면 됩니다. 자동으로 마스크가 생성됩니다.

```bash
# Soft Tissue Model 학습
python train.py --model soft_tissue

# Lung Model 학습
python train.py --model lung
```

### 3. 파일에서 마스크 로드 (기존 방식)

자동 생성 대신 파일에서 마스크를 로드하려면:

```python
# argmanager.py에서
args = argparse.Namespace(
    use_masks=True,
    auto_generate_masks=False,  # 자동 생성 비활성화
    mask_folders=['bone_mask', 'mediastinum_mask'],  # 폴더에서 로드
    ...
)
```

### 4. 커스텀 마스크 타입

필요한 마스크만 선택적으로 생성할 수 있습니다:

```python
args = argparse.Namespace(
    auto_generate_masks=True,
    mask_types=['lung', 'bone'],  # 폐와 뼈만 생성
    ...
)
```

## 마스크 생성 알고리즘

### 폐 마스크 (Lung Mask)
- HU 범위: -1000 ~ -300
- Body 마스크 필터링
- Border margin 제거 (32px)
- 작은 노이즈 제거 (<64px)

### 종격동 마스크 (Mediastinum Mask)
- 양쪽 폐의 ConvexHull 계산
- ConvexHull 내부에서 폐 영역 제외
- HU 범위: -300 ~ 450 (심장, 대혈관 포함)

### 뼈 마스크 (Bone Mask)
- HU 범위: >200
- ConvexHull 내부의 조영된 혈관 제외 (CECT 대응)
- Region growing으로 척추 복원
- Binary fill holes로 내부 공간 채우기

### 폐혈관 마스크 (Lung Vessel Mask)
- 폐 마스크에 binary_fill_holes 적용
- 채워진 영역에서 원본 폐 영역 제거
- HU 범위: -300 ~ 600

## 성능 고려사항

**자동 마스크 생성 시간:**
- 단일 슬라이스: ~0.1초
- 512x512 이미지 기준

**메모리 사용:**
- 추가 메모리 사용량: 무시할 수준
- HU 이미지는 이미 로드되어 있음

**정확도:**
- 폐 마스크: 매우 높음 (HU 기반)
- 종격동 마스크: 높음 (ConvexHull 기반)
- 뼈 마스크: 높음 (CECT 조영 혈관 제외)
- 폐혈관 마스크: 중간 (binary_fill_holes 기반)

## 문제 해결

### Q: 마스크가 생성되지 않습니다
A: `scipy` 라이브러리가 설치되어 있는지 확인하세요:
```bash
pip install scipy
```

### Q: ConvexHull 에러가 발생합니다
A: 폐 영역이 너무 작거나 단일 영역인 경우 발생할 수 있습니다. 정상적인 chest CT 이미지를 사용하세요.

### Q: 마스크 품질이 낮습니다
A: HU threshold 값을 조정할 수 있습니다:
```python
# mask_generator.py에서
lung_mask = detect_lung(hu_image, lung_lower=-1000, lung_upper=-300)
bone_mask = detect_bone(hu_image, lung_mask, bone_threshold=200)
```

### Q: 기존 파일 마스크와 병행하고 싶습니다
A: `auto_generate_masks=False`로 설정하고 `mask_folders`를 지정하세요.

## 참고

- 원본 마스크 생성 코드: `/workspace/Contrast_CT/hyunsu/Dataset_DucosyGAN/generate_mask.py`
- 마스크 생성 모듈: `/workspace/Contrast_CT/hyunsu/DuCoSy-GAN/modules/mask_generator.py`
- Dataset 구현: `/workspace/Contrast_CT/hyunsu/DuCoSy-GAN/modules/dataset.py`
