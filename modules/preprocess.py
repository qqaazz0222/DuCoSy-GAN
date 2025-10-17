import numpy as np
import torch
import pydicom

# ---- DICOM 이미지 전처리 ----
def apply_hu_transform(dicom_img, hu_min, hu_max):
    """ DICOM 이미지를 HU 값으로 변환하고 정규화 """
    image = dicom_img.pixel_array.astype(np.float32)
    image = image * float(dicom_img.RescaleSlope) + float(dicom_img.RescaleIntercept)
    image = np.clip(image, hu_min, hu_max)
    image = 2 * (image - hu_min) / (hu_max - hu_min) - 1
    return image


def apply_windowing(tensor_img, args):
    """ 모델 출력 텐서에 윈도잉 적용 """
    hu_img = (tensor_img + 1.0) / 2.0 * (args.hu_max - args.hu_min) + args.hu_min
    wc, ww = args.window_center, args.window_width
    hu_min_win, hu_max_win = wc - ww / 2.0, wc + ww / 2.0
    windowed_img = torch.clamp(hu_img, hu_min_win, hu_max_win)
    windowed_img = (windowed_img - hu_min_win) / ww
    return windowed_img


def preprocess_dicom(dcm_path, soft_tissue_hu_min, soft_tissue_hu_max, lung_hu_min, lung_hu_max):
    """DICOM 파일을 읽어 모델 입력에 맞게 전처리합니다."""
    dcm = pydicom.dcmread(dcm_path)
    
    # 1. HU 값으로 변환
    image = dcm.pixel_array.astype(np.float32)
    slope = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    image = image * slope + intercept
    
    # 2. HU 값 클리핑
    soft_tissue_image = np.clip(image, soft_tissue_hu_min, soft_tissue_hu_max)
    lung_image = np.clip(image, lung_hu_min, lung_hu_max)

    # 3. [-1, 1] 범위로 정규화
    soft_tissue_image = 2 * (soft_tissue_image - soft_tissue_hu_min) / (soft_tissue_hu_max - soft_tissue_hu_min) - 1
    lung_image = 2 * (lung_image - lung_hu_min) / (lung_hu_max - lung_hu_min) - 1

    # 4. 텐서로 변환 및 차원 추가 (C, H, W)
    soft_tissue_tensor = torch.from_numpy(soft_tissue_image).unsqueeze(0)
    lung_tensor = torch.from_numpy(lung_image).unsqueeze(0)

    return soft_tissue_tensor, lung_tensor, dcm


def postprocess_tensor(output_tensor, original_dcm, hu_min, hu_max):
    """모델 출력을 받아 DICOM 저장을 위한 pixel array로 변환합니다."""
    # 1. NumPy 배열로 변환
    output_array = output_tensor.cpu().squeeze().numpy()
    
    # 2. HU 값으로 역정규화 ([-1, 1] -> [hu_min, hu_max])
    denormalized_hu = (output_array + 1.0) / 2.0 * (hu_max - hu_min) + hu_min
    
    # 3. HU 값을 저장될 raw pixel 값으로 변환
    slope = float(original_dcm.RescaleSlope)
    intercept = float(original_dcm.RescaleIntercept)
    
    # DICOM 표준에 따른 역변환: StoredValue = (HU - RescaleIntercept) / RescaleSlope
    new_pixel_data = (denormalized_hu - intercept) / slope
    
    # 4. 원본 DICOM의 데이터 타입으로 변환 (e.g., int16)
    new_pixel_data = new_pixel_data.astype(original_dcm.pixel_array.dtype)
    
    return new_pixel_data