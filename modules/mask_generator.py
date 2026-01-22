"""
Anatomical Mask Generator Module
DICOM HU 값 기반 해부학적 마스크 생성 (폐, 종격동, 뼈, 폐혈관)
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path


def detect_lung(hu_volume, lung_lower=-1000, lung_upper=-300, min_size=64, border_margin=32):
    """HU 값 기반으로 폐 영역 마스크 생성"""
    # Body 마스크 생성
    body_mask = (hu_volume > -1000).astype(np.uint8)
    
    # Lung HU 조건
    lung_hu_mask = np.logical_and(hu_volume >= lung_lower, hu_volume <= lung_upper).astype(np.uint8)
    
    # Body 영역 내의 Lung만 선택
    lung_mask = np.logical_and(lung_hu_mask, body_mask).astype(np.uint8)
    
    # 2D 이미지인 경우 (단일 슬라이스)
    if lung_mask.ndim == 2:
        height, width = lung_mask.shape
        # Border margin 제거
        lung_mask[:border_margin, :] = 0
        lung_mask[height-border_margin:, :] = 0
        lung_mask[:, :border_margin] = 0
        lung_mask[:, width-border_margin:] = 0
        
        # 노이즈 제거
        labeled_slice, num_features = ndimage.label(lung_mask)
        for region_id in range(1, num_features + 1):
            region_size = (labeled_slice == region_id).sum()
            if region_size < min_size:
                lung_mask[labeled_slice == region_id] = 0
    else:
        # 3D 볼륨인 경우
        _, height, width = lung_mask.shape
        lung_mask[:, :border_margin, :] = 0
        lung_mask[:, height-border_margin:, :] = 0
        lung_mask[:, :, :border_margin] = 0
        lung_mask[:, :, width-border_margin:] = 0
        
        for z in range(lung_mask.shape[0]):
            labeled_slice, num_features = ndimage.label(lung_mask[z])
            for region_id in range(1, num_features + 1):
                region_size = (labeled_slice == region_id).sum()
                if region_size < min_size:
                    lung_mask[z][labeled_slice == region_id] = 0
    
    return lung_mask


def detect_lung_vessels(hu_volume, lung_mask, vessel_lower=-300, vessel_upper=600):
    """폐 내부 혈관 영역 감지"""
    body_mask = (hu_volume > -1000).astype(np.uint8)
    
    # 2D 이미지인 경우
    if lung_mask.ndim == 2:
        lung_slice = lung_mask
        body_slice = body_mask
        
        labeled_lung, num_lung_regions = ndimage.label(lung_slice)
        body_area = body_slice.sum()
        lung_area = lung_slice.sum()
        
        if num_lung_regions >= 2 and body_area > 0 and (lung_area / body_area) >= 0.1:
            filled_lung = ndimage.binary_fill_holes(lung_slice).astype(np.uint8)
            vessel_candidate = filled_lung - lung_slice
        else:
            vessel_candidate = np.zeros_like(lung_slice)
        
        hu_condition = np.logical_and(hu_volume >= vessel_lower, hu_volume <= vessel_upper)
        vessel_mask = np.logical_and(vessel_candidate, hu_condition).astype(np.uint8)
        return vessel_mask
    
    # 3D 볼륨인 경우
    vessel_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    
    for z in range(lung_mask.shape[0]):
        lung_slice = lung_mask[z]
        body_slice = body_mask[z] if body_mask.ndim == 3 else body_mask
        
        labeled_lung, num_lung_regions = ndimage.label(lung_slice)
        body_area = body_slice.sum()
        lung_area = lung_slice.sum()
        
        if num_lung_regions >= 2 and body_area > 0 and (lung_area / body_area) >= 0.1:
            filled_lung = ndimage.binary_fill_holes(lung_slice).astype(np.uint8)
            vessel_candidate = filled_lung - lung_slice
        else:
            vessel_candidate = np.zeros_like(lung_slice)
        
        hu_slice = hu_volume[z] if hu_volume.ndim == 3 else hu_volume
        hu_condition = np.logical_and(hu_slice >= vessel_lower, hu_slice <= vessel_upper)
        vessel_mask[z] = np.logical_and(vessel_candidate, hu_condition).astype(np.uint8)
    
    return vessel_mask


def detect_mediastinum(hu_volume, lung_mask, mediastinum_lower=-300, mediastinum_upper=450):
    """종격동(Mediastinum) 영역 마스크 생성"""
    body_mask = (hu_volume > -1000).astype(np.uint8)
    
    # 2D 이미지인 경우
    if lung_mask.ndim == 2:
        lung_slice = lung_mask
        body_slice = body_mask
        hu_slice = hu_volume
        
        labeled_lung, num_lung_regions = ndimage.label(lung_slice)
        body_area = body_slice.sum()
        lung_area = lung_slice.sum()
        
        if num_lung_regions >= 2 and body_area > 0 and (lung_area / body_area) >= 0.1:
            lung_coords = np.argwhere(lung_slice == 1)
            
            if len(lung_coords) >= 3:
                try:
                    hull = ConvexHull(lung_coords)
                    hull_path = Path(lung_coords[hull.vertices])
                    
                    y_coords, x_coords = np.mgrid[:lung_slice.shape[0], :lung_slice.shape[1]]
                    points = np.vstack((y_coords.flatten(), x_coords.flatten())).T
                    convex_hull = hull_path.contains_points(points).reshape(lung_slice.shape).astype(np.uint8)
                except:
                    convex_hull = lung_slice.copy()
            else:
                convex_hull = lung_slice.copy()
            
            mediastinum_candidate = convex_hull - lung_slice
            hu_condition = np.logical_and(hu_slice >= mediastinum_lower, hu_slice <= mediastinum_upper)
            mediastinum_mask = np.logical_and(mediastinum_candidate, hu_condition).astype(np.uint8)
        else:
            mediastinum_mask = np.zeros_like(lung_slice)
        
        return mediastinum_mask
    
    # 3D 볼륨인 경우
    mediastinum_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    
    for z in range(lung_mask.shape[0]):
        lung_slice = lung_mask[z]
        body_slice = body_mask[z] if body_mask.ndim == 3 else body_mask
        hu_slice = hu_volume[z] if hu_volume.ndim == 3 else hu_volume
        
        labeled_lung, num_lung_regions = ndimage.label(lung_slice)
        body_area = body_slice.sum()
        lung_area = lung_slice.sum()
        
        if num_lung_regions >= 2 and body_area > 0 and (lung_area / body_area) >= 0.1:
            lung_coords = np.argwhere(lung_slice == 1)
            
            if len(lung_coords) >= 3:
                try:
                    hull = ConvexHull(lung_coords)
                    hull_path = Path(lung_coords[hull.vertices])
                    
                    y_coords, x_coords = np.mgrid[:lung_slice.shape[0], :lung_slice.shape[1]]
                    points = np.vstack((y_coords.flatten(), x_coords.flatten())).T
                    convex_hull = hull_path.contains_points(points).reshape(lung_slice.shape).astype(np.uint8)
                except:
                    convex_hull = lung_slice.copy()
            else:
                convex_hull = lung_slice.copy()
            
            mediastinum_candidate = convex_hull - lung_slice
            hu_condition = np.logical_and(hu_slice >= mediastinum_lower, hu_slice <= mediastinum_upper)
            mediastinum_mask[z] = np.logical_and(mediastinum_candidate, hu_condition).astype(np.uint8)
        else:
            mediastinum_mask[z] = np.zeros_like(lung_slice)
    
    return mediastinum_mask


def detect_bone(hu_volume, lung_mask, bone_threshold=200, spine_margin_ratio=0.25):
    """HU 값 기반으로 뼈 영역 마스크 생성 (CECT 대응 포함)"""
    body_mask = (hu_volume > -1000).astype(np.uint8)
    
    # 전체 뼈 후보
    all_bone_candidate = (hu_volume >= bone_threshold).astype(np.uint8)
    all_bone_candidate = np.logical_and(all_bone_candidate, body_mask).astype(np.uint8)
    
    bone_mask = all_bone_candidate.copy()
    
    # 2D 이미지인 경우
    if lung_mask.ndim == 2:
        lung_slice = lung_mask
        body_slice = body_mask
        
        labeled_lung, num_lung_regions = ndimage.label(lung_slice)
        body_area = body_slice.sum()
        lung_area = lung_slice.sum()
        
        if num_lung_regions >= 2 and body_area > 0 and (lung_area / body_area) >= 0.1:
            lung_coords = np.argwhere(lung_slice == 1)
            
            if len(lung_coords) >= 3:
                try:
                    hull = ConvexHull(lung_coords)
                    hull_path = Path(lung_coords[hull.vertices])
                    
                    height, width = lung_slice.shape
                    y_coords, x_coords = np.mgrid[:height, :width]
                    points = np.vstack((y_coords.flatten(), x_coords.flatten())).T
                    
                    convex_hull_mask = hull_path.contains_points(points).reshape(lung_slice.shape).astype(np.uint8)
                    
                    # 척추 영역 보존
                    spine_region_mask = np.zeros_like(lung_slice, dtype=np.uint8)
                    spine_start = int(height * (1 - spine_margin_ratio))
                    spine_region_mask[spine_start:, :] = 1
                    
                    # 종격동 혈관 영역 (제외할 부분)
                    mediastinum_vessel_region = convex_hull_mask.copy()
                    mediastinum_vessel_region = np.logical_and(mediastinum_vessel_region, 1 - lung_slice)
                    mediastinum_vessel_region = np.logical_and(mediastinum_vessel_region, 1 - spine_region_mask)
                    
                    bone_mask = np.logical_and(bone_mask, 1 - mediastinum_vessel_region).astype(np.uint8)
                except:
                    pass
        
        # Region growing으로 척추 복원
        all_bone_slice = all_bone_candidate
        removed_bone = np.logical_and(all_bone_slice, 1 - bone_mask).astype(np.uint8)
        
        if removed_bone.sum() > 0:
            combined = np.logical_or(bone_mask, removed_bone).astype(np.uint8)
            labeled_combined, num_features = ndimage.label(combined)
            
            bone_labels = set(np.unique(labeled_combined[bone_mask > 0]))
            bone_labels.discard(0)
            
            for label_id in bone_labels:
                label_mask = (labeled_combined == label_id)
                hu_valid = (hu_volume >= bone_threshold)
                valid_addition = np.logical_and(label_mask, hu_valid).astype(np.uint8)
                bone_mask = np.logical_or(bone_mask, valid_addition).astype(np.uint8)
        
        # Fill holes
        if bone_mask.sum() > 0:
            bone_mask = ndimage.binary_fill_holes(bone_mask).astype(np.uint8)
        
        return bone_mask
    
    # 3D 볼륨인 경우
    for z in range(lung_mask.shape[0]):
        lung_slice = lung_mask[z]
        body_slice = body_mask[z] if body_mask.ndim == 3 else body_mask
        
        labeled_lung, num_lung_regions = ndimage.label(lung_slice)
        body_area = body_slice.sum()
        lung_area = lung_slice.sum()
        
        if num_lung_regions >= 2 and body_area > 0 and (lung_area / body_area) >= 0.1:
            lung_coords = np.argwhere(lung_slice == 1)
            
            if len(lung_coords) >= 3:
                try:
                    hull = ConvexHull(lung_coords)
                    hull_path = Path(lung_coords[hull.vertices])
                    
                    height, width = lung_slice.shape
                    y_coords, x_coords = np.mgrid[:height, :width]
                    points = np.vstack((y_coords.flatten(), x_coords.flatten())).T
                    
                    convex_hull_mask = hull_path.contains_points(points).reshape(lung_slice.shape).astype(np.uint8)
                    
                    spine_region_mask = np.zeros_like(lung_slice, dtype=np.uint8)
                    spine_start = int(height * (1 - spine_margin_ratio))
                    spine_region_mask[spine_start:, :] = 1
                    
                    mediastinum_vessel_region = convex_hull_mask.copy()
                    mediastinum_vessel_region = np.logical_and(mediastinum_vessel_region, 1 - lung_slice)
                    mediastinum_vessel_region = np.logical_and(mediastinum_vessel_region, 1 - spine_region_mask)
                    
                    bone_mask[z] = np.logical_and(bone_mask[z], 1 - mediastinum_vessel_region).astype(np.uint8)
                except:
                    pass
    
    # Region growing
    for z in range(bone_mask.shape[0]):
        bone_slice = bone_mask[z]
        all_bone_slice = all_bone_candidate[z] if all_bone_candidate.ndim == 3 else all_bone_candidate
        hu_slice = hu_volume[z] if hu_volume.ndim == 3 else hu_volume
        
        removed_bone = np.logical_and(all_bone_slice, 1 - bone_slice).astype(np.uint8)
        
        if removed_bone.sum() == 0:
            continue
        
        combined = np.logical_or(bone_slice, removed_bone).astype(np.uint8)
        labeled_combined, num_features = ndimage.label(combined)
        
        bone_labels = set(np.unique(labeled_combined[bone_slice > 0]))
        bone_labels.discard(0)
        
        for label_id in bone_labels:
            label_mask = (labeled_combined == label_id)
            hu_valid = (hu_slice >= bone_threshold)
            valid_addition = np.logical_and(label_mask, hu_valid).astype(np.uint8)
            bone_mask[z] = np.logical_or(bone_mask[z], valid_addition).astype(np.uint8)
    
    # Fill holes
    for z in range(bone_mask.shape[0]):
        if bone_mask[z].sum() > 0:
            bone_mask[z] = ndimage.binary_fill_holes(bone_mask[z]).astype(np.uint8)
    
    return bone_mask


def generate_anatomical_masks(hu_image, mask_types=['lung', 'mediastinum', 'bone', 'lung_vessel']):
    """
    HU 이미지로부터 해부학적 마스크 생성
    
    Args:
        hu_image: HU 값의 2D 또는 3D numpy 배열
        mask_types: 생성할 마스크 종류 리스트
    
    Returns:
        dict: {mask_name: mask_array} 형태의 딕셔너리
    """
    masks = {}
    
    # 폐 마스크 (필수)
    if 'lung' in mask_types or 'lung_vessel' in mask_types or 'mediastinum' in mask_types or 'bone' in mask_types:
        lung_mask = detect_lung(hu_image)
        if 'lung' in mask_types:
            masks['lung'] = lung_mask
    
    # 종격동 마스크
    if 'mediastinum' in mask_types:
        mediastinum_mask = detect_mediastinum(hu_image, lung_mask)
        masks['mediastinum'] = mediastinum_mask
    
    # 뼈 마스크
    if 'bone' in mask_types:
        bone_mask = detect_bone(hu_image, lung_mask)
        masks['bone'] = bone_mask
    
    # 폐혈관 마스크
    if 'lung_vessel' in mask_types:
        lung_vessel_mask = detect_lung_vessels(hu_image, lung_mask)
        masks['lung_vessel'] = lung_vessel_mask
    
    return masks
