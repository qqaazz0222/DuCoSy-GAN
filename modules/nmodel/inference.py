#!/usr/bin/env python3
"""
Simple Inference Script for U-Net 3D Model
"""
import os
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
import cv2

from modules.nmodel.model import UNet3D, UNet3DLight
from modules.nmodel.dataset import CTDiffDataset
from modules.nmodel.config import Config


def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 체크포인트에 down4/up4가 있는지 확인하여 모델 클래스 결정
    state_dict = checkpoint['model_state_dict']
    has_down4 = any('down4' in key for key in state_dict.keys())
    use_full_model = has_down4
    
    # 체크포인트에서 모델 설정 추출
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        base_channels = saved_config.get('base_channels', 16)
        in_channels = saved_config.get('in_channels', 1)
        out_channels = saved_config.get('out_channels', 1)
    else:
        # 체크포인트에 설정이 없으면 state_dict에서 추론
        first_conv_weight = state_dict['inc.double_conv.0.weight']
        base_channels = first_conv_weight.shape[0]  # 출력 채널 = base_channels
        in_channels = first_conv_weight.shape[1]
        out_channels = 1
    
    # 올바른 모델 클래스 선택
    if use_full_model:
        model = UNet3D(
            n_channels=in_channels,
            n_classes=out_channels,
            base_channels=base_channels
        ).to(device)
    else:
        model = UNet3DLight(
            n_channels=in_channels,
            n_classes=out_channels,
            base_channels=base_channels
        ).to(device)
    
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Config 객체 생성 (정규화 함수 사용을 위해)
    config = Config()
    config.base_channels = base_channels
    config.in_channels = in_channels
    config.out_channels = out_channels
    
    return model, config


def predict_volume(model, vue_volume, device='cuda', use_amp=True):
    d, h, w = vue_volume.shape
    output_volume = np.zeros((d, h, w), dtype=np.float32)
    input_normalized = CTDiffDataset.normalize_hu(vue_volume)
    
    with torch.no_grad():
        for slice_idx in range(d):
            slice_data = input_normalized[slice_idx:slice_idx+1, :, :]
            slice_tensor = torch.from_numpy(slice_data).float()
            slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            with autocast('cuda', enabled=use_amp):
                slice_output = model(slice_tensor)
            
            output_volume[slice_idx, :, :] = slice_output.squeeze().cpu().numpy()
    
    predicted_diff = CTDiffDataset.denormalize_diff(output_volume)
    return predicted_diff


def save_results(predicted_diff, output_dir, base_name, original_volume=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predicted difference map
    volume_path = os.path.join(output_dir, f'{base_name}_predicted_diff.npy')
    np.save(volume_path, predicted_diff)
    print(f"✓ Predicted diff volume saved: {volume_path}")
    
    # Save combined volume (original + predicted_diff)
    if original_volume is not None:
        combined_volume = original_volume + predicted_diff
        combined_path = os.path.join(output_dir, f'{base_name}_combined.npy')
        np.save(combined_path, combined_volume)
        print(f"✓ Combined volume saved: {combined_path}")
        print(f"  Original range: [{original_volume.min():.2f}, {original_volume.max():.2f}]")
        print(f"  Combined range: [{combined_volume.min():.2f}, {combined_volume.max():.2f}]")
    
    # Save slices (difference map)
    slice_dir = os.path.join(output_dir, f'{base_name}_diff_slices')
    os.makedirs(slice_dir, exist_ok=True)
    
    global_min, global_max = predicted_diff.min(), predicted_diff.max()
    d = predicted_diff.shape[0]
    
    for i in tqdm(range(d), desc='Saving diff slices'):
        slice_data = predicted_diff[i]
        if global_max > global_min:
            normalized = ((slice_data - global_min) / (global_max - global_min) * 255)
        else:
            normalized = np.full_like(slice_data, 128)
        image = normalized.astype(np.uint8)
        cv2.imwrite(os.path.join(slice_dir, f'slice_{i:04d}.png'), image)
    
    print(f"✓ {d} diff slices saved: {slice_dir}")
    
    # Save combined slices if original volume is provided
    if original_volume is not None:
        combined_slice_dir = os.path.join(output_dir, f'{base_name}_combined_slices')
        os.makedirs(combined_slice_dir, exist_ok=True)
        
        combined_min, combined_max = combined_volume.min(), combined_volume.max()
        
        for i in tqdm(range(d), desc='Saving combined slices'):
            slice_data = combined_volume[i]
            if combined_max > combined_min:
                normalized = ((slice_data - combined_min) / (combined_max - combined_min) * 255)
            else:
                normalized = np.full_like(slice_data, 128)
            image = normalized.astype(np.uint8)
            cv2.imwrite(os.path.join(combined_slice_dir, f'slice_{i:04d}.png'), image)
        
        print(f"✓ {d} combined slices saved: {combined_slice_dir}")
        
        
def nomalize_volume(volume):
    global_min, global_max = volume.min(), volume.max()
    d = volume.shape[0]
    
    nomalized_volume = []
    
    for i in range(d):
        slice_data = volume[i]
        if global_max > global_min:
            normalized = ((slice_data - global_min) / (global_max - global_min) * 255)
        else:
            normalized = np.full_like(slice_data, 128)
        nomalized_volume.append(normalized.astype(np.uint8))

    return np.array(nomalized_volume)