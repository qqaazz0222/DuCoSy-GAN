import os
import numpy as np
import pydicom

file1 = "/workspace/ct-dual-energy/DuCoSy-GAN/data/input/Kyunghee_Univ/KP-0007/POST STD/1.2.410.200058.2.2.82.0895.1.108296.dcm"
# file2 = "/workspace/ct-dual-energy/DuCoSy-GAN/data/input/Kangwon_National_Univ/00000102-1542-11-06/POST STD/anonymized_00000102_800.dcm"
file2 = "/workspace/ct-dual-energy/DuCoSy-GAN/data/input/Kyunghee_Univ/KP-0007/POST VUE/1.2.410.200058.2.2.82.0895.1.108603.dcm"
dcm1 = pydicom.dcmread(file1)
dcm2 = pydicom.dcmread(file2)
# print(dcm1)
print(dcm2)