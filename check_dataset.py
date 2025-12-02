import os
import csv
import glob
import pydicom

dataset_dir = '/workspace/ct-dual-energy/DuCoSy-GAN/data/input/Kyunghee_Univ'

def check(dataset_dir, csv_path="checked_list.csv"):
    """ 데이터셋 내 각 환자의 NCCT와 CECT 슬라이스 위치 일치 여부 확인 """
    dataset_name = os.path.basename(dataset_dir)

    patient_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, '*')) if os.path.isdir(d)])

    error_list = []
    
    if not os.path.exists(csv_path):
        headers = ['Dataset', 'Patient ID', 'NCCT Min Z', 'NCCT Max Z', 'CECT Min Z', 'CECT Max Z', 'Flag']
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        ncct_dir = os.path.join(patient_dir, 'POST VUE')
        cect_dir = os.path.join(patient_dir, 'POST STD')
        ncct_list = sorted(glob.glob(os.path.join(ncct_dir, '*.dcm')))
        cect_list = sorted(glob.glob(os.path.join(cect_dir, '*.dcm')))
        
        ncct_location_list = []
        cect_location_list = []
        
        for idx, dcm_path in enumerate(ncct_list):
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            z_location = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
            ncct_location_list.append((z_location))
        
        for idx, dcm_path in enumerate(cect_list):
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            z_location = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
            cect_location_list.append((z_location))
            
        for ncct_z_loc, cect_z_loc in zip(sorted(ncct_location_list), sorted(cect_location_list)):
            if ncct_z_loc != cect_z_loc:
                if patient_id not in error_list:
                    error_list.append(patient_id)
                
        print(f"\nPatient ID: {patient_id}")
        print(f"- NCCT: min_z_loc={min(ncct_location_list)}, max_z_loc={max(ncct_location_list)}")
        print(f"- CECT: min_z_loc={min(cect_location_list)}, max_z_loc={max(cect_location_list)}")
        flag = 'Mismatch' if patient_id in error_list else 'Match'
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name, patient_id, min(ncct_location_list), max(ncct_location_list), 
                             min(cect_location_list), max(cect_location_list), flag])
            
    print("Patients with mismatched slice locations between NCCT and CECT:")
    for patient_id in error_list:
        print(patient_id)
    print(f"Total: {len(error_list)} patients")
    
if __name__ == "__main__":
    dataset_dir = 'data/input/Kangwon_National_Univ'
    check(dataset_dir, csv_path="checked_list_kangwon.csv")
    dataset_dir = 'data/input/Kyunghee_Univ'
    check(dataset_dir, csv_path="checked_list_kyunghee.csv")
    print("Checking completed.")