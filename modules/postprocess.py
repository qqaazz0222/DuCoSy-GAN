import os
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, gaussian_filter
from scipy.interpolate import interp1d

def postprocess_ct_volume(volume, method='gaussian3d', enhance_sharpness=True, hu_threshold=750, **kwargs):
    """
    CT 볼륨 데이터를 불러와서 후처리(슬라이스 간 연속성 개선 및 선명도 향상)를 적용합니다.
    
    Parameters:
    -----------
    raw_file_path : str
        원본 .npy 파일 경로
    output_file_path : str
        저장할 후처리된 파일 경로
    method : str
        스무딩 방법 선택 ('gaussian', 'gaussian3d', 'adaptive', 'median', 'interpolation', 'kalman')
        - 'gaussian': 가우시안 필터 (sigma 파라미터 조절 가능)
        - 'gaussian3d': 3D 가우시안 필터 (계단 현상 제거에 가장 효과적, 기본값)
        - 'adaptive': 적응형 스무딩 (슬라이스 간 차이가 큰 곳에 더 강한 스무딩)
        - 'median': 메디안 필터 (kernel_size 파라미터 조절 가능)
        - 'interpolation': 큐빅 스플라인 보간
        - 'kalman': 칼만 필터 (process_variance, measurement_variance 조절 가능)
    enhance_sharpness : bool
        스무딩 후 선명도 향상 적용 여부 (기본값=True)
    hu_threshold : float
        HU 임계값 (기본값=750). 이 값 이상의 복셀(뼈 등)에는 후처리를 적용하지 않음
    **kwargs : dict
        각 메서드별 추가 파라미터
        - gaussian: sigma (기본값=1.0)
        - gaussian3d: sigma_z (기본값=2.0), sigma_xy (기본값=0.5)
        - adaptive: base_sigma (기본값=1.5), max_sigma (기본값=3.0)
        - median: kernel_size (기본값=3)
        - kalman: process_variance (기본값=1e-5), measurement_variance (기본값=1e-2)
        - sharpen_amount (기본값=0.5): 선명도 강도 (0.0~1.0)
        - sharpen_radius (기본값=1.0): 선명화 반경
    
    Returns:
    --------
    postprocessed_volume : numpy.ndarray
        후처리된 볼륨 데이터
    """
    original_volume = volume.copy()  # 원본 저장
    
    # HU 임계값 이상의 영역 마스크 생성 (뼈 등 고밀도 조직)
    high_density_mask = volume >= hu_threshold
    
    # 스무딩 적용
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        postprocessed_volume = gaussian_filter1d(volume, sigma=sigma, axis=0)
        
    elif method == 'gaussian3d':
        sigma_z = kwargs.get('sigma_z', 2.0)
        sigma_xy = kwargs.get('sigma_xy', 0.5)
        postprocessed_volume = gaussian_filter(volume, sigma=(sigma_z, sigma_xy, sigma_xy))
        
    elif method == 'adaptive':
        base_sigma = kwargs.get('base_sigma', 1.5)
        max_sigma = kwargs.get('max_sigma', 3.0)
        postprocessed_volume = adaptive_smooth(volume, base_sigma, max_sigma)
        
    elif method == 'median':
        kernel_size = kwargs.get('kernel_size', 3)
        postprocessed_volume = median_filter(volume, size=(kernel_size, 1, 1))
        
    elif method == 'interpolation':
        num_slices = volume.shape[0]
        # 원본 슬라이스 인덱스
        original_indices = np.arange(num_slices)
        # 보간을 위한 더 세밀한 인덱스 (2배 해상도)
        fine_indices = np.linspace(0, num_slices - 1, num_slices * 2)
        
        postprocessed_volume = np.zeros((num_slices, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
        
        # 각 픽셀 위치에 대해 z축 방향으로 보간
        for i in range(volume.shape[1]):
            for j in range(volume.shape[2]):
                pixel_series = volume[:, i, j]
                # 큐빅 스플라인 보간
                interpolator = interp1d(original_indices, pixel_series, kind='cubic', fill_value='extrapolate')
                # 보간 후 다시 원래 슬라이스 수로 샘플링
                fine_series = interpolator(fine_indices)
                postprocessed_volume[:, i, j] = fine_series[::2]  # 원래 슬라이스 수로 다운샘플링
        
    elif method == 'kalman':
        process_variance = kwargs.get('process_variance', 1e-5)
        measurement_variance = kwargs.get('measurement_variance', 1e-2)
        postprocessed_volume = apply_kalman_filter(volume, process_variance, measurement_variance)
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'gaussian', 'gaussian3d', 'adaptive', 'median', 'interpolation', 'kalman'")
    
    # 선명도 향상 적용
    if enhance_sharpness:
        sharpen_amount = kwargs.get('sharpen_amount', 0.5)
        sharpen_radius = kwargs.get('sharpen_radius', 1.0)
        postprocessed_volume = unsharp_mask(postprocessed_volume, original_volume, 
                                       amount=sharpen_amount, radius=sharpen_radius)
    
    # HU 임계값 이상의 영역은 원본 값으로 복원 (뼈 등 고밀도 조직 보존)
    postprocessed_volume[high_density_mask] = original_volume[high_density_mask]
    
    # 데이터 타입 유지
    postprocessed_volume = postprocessed_volume.astype(np.int16)
    
    return postprocessed_volume


def unsharp_mask(smoothed_volume, original_volume, amount=0.5, radius=1.0):
    """
    Unsharp Masking을 적용하여 스무딩된 볼륨의 선명도를 향상시킵니다.
    
    Parameters:
    -----------
    smoothed_volume : numpy.ndarray
        스무딩된 볼륨
    original_volume : numpy.ndarray
        원본 볼륨
    amount : float
        선명화 강도 (0.0~1.0, 기본값=0.5)
        - 0.0: 선명화 없음
        - 0.5: 적당한 선명화 (권장)
        - 1.0: 강한 선명화
    radius : float
        선명화 반경 (기본값=1.0)
        - 작을수록: 세밀한 디테일 강조
        - 클수록: 넓은 영역의 경계 강조
    
    Returns:
    --------
    sharpened : numpy.ndarray
        선명도가 향상된 볼륨
    """
    smoothed_volume = smoothed_volume.astype(np.float64)
    original_volume = original_volume.astype(np.float64)
    
    # 고주파 성분 추출 (원본의 디테일)
    # xy 평면에서만 선명화 (z축은 이미 스무딩됨)
    blurred = gaussian_filter(smoothed_volume, sigma=(0, radius, radius))
    high_freq = smoothed_volume - blurred
    
    # 원본의 고주파 성분도 일부 활용
    original_blurred = gaussian_filter(original_volume, sigma=(0, radius, radius))
    original_high_freq = original_volume - original_blurred
    
    # 두 고주파 성분을 혼합하여 적용
    combined_high_freq = (1 - amount) * high_freq + amount * original_high_freq
    
    # 선명화 적용
    sharpened = smoothed_volume + combined_high_freq * amount
    
    # 값 범위 클리핑 (원본 범위 유지)
    sharpened = np.clip(sharpened, original_volume.min(), original_volume.max())
    
    return sharpened


def adaptive_smooth(volume, base_sigma=1.5, max_sigma=3.0):
    """
    적응형 스무딩: 슬라이스 간 차이가 큰 곳에 더 강한 스무딩을 적용합니다.
    
    Parameters:
    -----------
    volume : numpy.ndarray
        입력 볼륨 (slices, height, width)
    base_sigma : float
        기본 시그마 값
    max_sigma : float
        최대 시그마 값
    
    Returns:
    --------
    smoothed_volume : numpy.ndarray
        스무딩된 볼륨
    """
    num_slices = volume.shape[0]
    
    # 1단계: 슬라이스 간 차이 계산
    slice_diffs = np.zeros(num_slices - 1)
    for i in range(num_slices - 1):
        diff = np.abs(volume[i+1] - volume[i])
        slice_diffs[i] = np.mean(diff)
    
    # 2단계: 차이에 기반한 적응형 sigma 계산
    max_diff = np.max(slice_diffs) if np.max(slice_diffs) > 0 else 1.0
    
    # 3단계: 각 슬라이스에 적응형 스무딩 적용
    smoothed_volume = volume.copy().astype(np.float64)
    
    # z축 방향으로 강한 1차 스무딩
    smoothed_volume = gaussian_filter1d(smoothed_volume, sigma=base_sigma, axis=0)
    
    # xy 평면에서 경계를 부드럽게 (계단 현상 제거)
    smoothed_volume = gaussian_filter(smoothed_volume, sigma=(max_sigma, 0.3, 0.3))
    
    return smoothed_volume


def apply_kalman_filter(volume, process_variance=1e-5, measurement_variance=1e-2):
    """
    1D 칼만 필터를 z축 방향으로 적용하여 슬라이스 간 연속성을 개선합니다.
    
    Parameters:
    -----------
    volume : numpy.ndarray
        입력 볼륨 (slices, height, width)
    process_variance : float
        프로세스 노이즈 분산 (Q)
    measurement_variance : float
        측정 노이즈 분산 (R)
    
    Returns:
    --------
    filtered_volume : numpy.ndarray
        필터링된 볼륨
    """
    num_slices, height, width = volume.shape
    filtered_volume = np.zeros_like(volume, dtype=np.float64)
    
    # 각 픽셀 위치에 대해 z축 방향으로 칼만 필터 적용
    for i in range(height):
        for j in range(width):
            pixel_series = volume[:, i, j].astype(np.float64)
            filtered_series = kalman_filter_1d(pixel_series, process_variance, measurement_variance)
            filtered_volume[:, i, j] = filtered_series
    
    return filtered_volume


def kalman_filter_1d(measurements, process_variance, measurement_variance):
    """
    1차원 칼만 필터 구현
    
    Parameters:
    -----------
    measurements : numpy.ndarray
        측정값 시계열
    process_variance : float
        프로세스 노이즈 분산 (Q)
    measurement_variance : float
        측정 노이즈 분산 (R)
    
    Returns:
    --------
    filtered : numpy.ndarray
        필터링된 값
    """
    n = len(measurements)
    filtered = np.zeros(n)
    
    # 초기화
    x_est = measurements[0]  # 초기 상태 추정
    p_est = 1.0  # 초기 오차 공분산
    
    for k in range(n):
        # 예측 단계
        x_pred = x_est
        p_pred = p_est + process_variance
        
        # 업데이트 단계
        K = p_pred / (p_pred + measurement_variance)  # 칼만 이득
        x_est = x_pred + K * (measurements[k] - x_pred)
        p_est = (1 - K) * p_pred
        
        filtered[k] = x_est
    
    return filtered


def apply_diffmap(volume, diff_volume, threshold=8):
    """
    주어진 차이 맵을 원본 볼륨에 적용하여 수정된 볼륨을 생성합니다.
    
    Parameters:
    -----------
    volume : numpy.ndarray
        원본 볼륨 데이터
    diff_map : numpy.ndarray
        차이 맵 데이터 (원본과 동일한 크기)
    
    Returns:
    --------
    modified_volume : numpy.ndarray
        차이 맵이 적용된 수정된 볼륨
    """
    if type(volume) != np.ndarray:
        volume = np.array(volume)
    if type(diff_volume) != np.ndarray:
        diff_volume = np.array(diff_volume)
        
    diff_volume[diff_volume < threshold] = 0
    diff_volume = diff_volume.astype(np.uint8)
    
    modified_volume = volume + diff_volume

    return modified_volume