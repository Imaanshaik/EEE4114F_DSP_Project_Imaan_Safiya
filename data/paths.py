import sys
from pathlib import Path

MAIN_DIR = Path(sys.argv[0]).resolve().parent   
MOTION_SENSE_PATH = MAIN_DIR.parent / "motion-sense"  
SUBJECTS_INFO_CSV_PATH = MOTION_SENSE_PATH / "data" / "data_subjects_info.csv"
A_DEVICE_MOTION_DATA_PATH = MOTION_SENSE_PATH / "data" / "A_DeviceMotion_data"
B_DEVICE_MOTION_DATA_PATH = MOTION_SENSE_PATH / "data" / "B_DeviceMotion_data"
C_DEVICE_MOTION_DATA_PATH = MOTION_SENSE_PATH / "data" / "C_DeviceMotion_data"
