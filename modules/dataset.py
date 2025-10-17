import os
import glob
import pydicom
from torch.utils.data import Dataset
from modules.preprocess import apply_hu_transform
from tqdm import tqdm
import warnings

# pydicom 라이브러리에서 발생하는 사용자 경고를 무시
warnings.filterwarnings("ignore", category=UserWarning)
      
        
# ---- 데이터셋 및 유틸리티 함수 -----
class DicomDataset(Dataset):
    """ DICOM 파일 쌍을 로드하는 커스텀 데이터셋 """
    def __init__(self, patient_dirs, args, transform=None):
        self.transform = transform
        self.args = args
        self.paired_files = []
        
        for patient_dir in tqdm(patient_dirs, desc="데이터 처리 중"):
            ncct_path = os.path.join(patient_dir, args.ncct_folder)
            cect_path = os.path.join(patient_dir, args.cect_folder)
            
            ncct_files = sorted(glob.glob(os.path.join(ncct_path, "*.dcm")))
            cect_files = sorted(glob.glob(os.path.join(cect_path, "*.dcm")))

            if not ncct_files or not cect_files: continue

            try:
                ncct_files.sort(key=lambda x: float(pydicom.dcmread(x, stop_before_pixels=True).SliceLocation))
                cect_files.sort(key=lambda x: float(pydicom.dcmread(x, stop_before_pixels=True).SliceLocation))
            except (AttributeError, KeyError):
                print(f"Warning: SliceLocation not found in {patient_dir}. Falling back to filename sort.")
                pass

            for ncct_file, cect_file in zip(ncct_files, cect_files):
                self.paired_files.append((ncct_file, cect_file))
                
    def __len__(self): return len(self.paired_files)
    def __getitem__(self, index):
        ncct_path, cect_path = self.paired_files[index]
        ncct_dcm, cect_dcm = pydicom.dcmread(ncct_path), pydicom.dcmread(cect_path)
        ncct_img = apply_hu_transform(ncct_dcm, self.args.hu_min, self.args.hu_max)
        cect_img = apply_hu_transform(cect_dcm, self.args.hu_min, self.args.hu_max)
        if self.transform:
            ncct_img, cect_img = self.transform(ncct_img), self.transform(cect_img)
        return {"A": ncct_img, "B": cect_img}


