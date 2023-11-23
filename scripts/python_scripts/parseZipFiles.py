import os
from collections.abc import Iterable
from datetime import datetime, timedelta
from zipfile import ZipFile

import numpy as np
from scipy.io import savemat

from src.utils.utils import stack_arrays

WORK_FOLDER = '/work/ivan.santos'
SCRATCH_FOLDER = '/scratch/ivan.santos'

root_folder = WORK_FOLDER + '/unprocessed_dataset/z24'
output_root_folder = WORK_FOLDER + '/datasets/z24'

day_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
}

INITIAL_DATE = datetime(1997, 11, 8)


def get_datetime(week_number: int, day_code: str, hour=0) -> datetime:
    day_offset = 7 * (week_number - 1) + day_map[day_code]
    return INITIAL_DATE + timedelta(days=day_offset, hours=hour)


def extract_acceleration_from_zip(zip_file: ZipFile, data_filename_to_extract: str) -> np.ndarray:
    with zip_file.open(data_filename_to_extract, 'r') as data_file:
        lines = data_file.readlines()
        number_of_samples = int(lines[1].decode('ascii'))
        initial_data_line = 3
        return np.array(list(map(lambda byte_line: float(byte_line.decode('ascii')),
                                 lines[initial_data_line: initial_data_line + number_of_samples])))


def process_zip(zip_path: str, zip_filename: str) -> np.ndarray:
    sensors_to_extract = ['05', '06', '07', '12', '14', '16']
    output_data = np.empty(0)

    with ZipFile(zip_path, 'r') as zip_file:
        zip_namelist = zip_file.namelist()

        for sensor_number in sensors_to_extract:
            data_filename_to_extract = zip_filename.split('.')[0] + sensor_number + '.aaa'

            if data_filename_to_extract in zip_namelist:
                acceleration_data = extract_acceleration_from_zip(zip_file, data_filename_to_extract)
                output_data = stack_arrays(output_data, acceleration_data)
            else:
                print(f"File {data_filename_to_extract} doesn't exist in the zip file ({zip_path})")

    return output_data.T


def create_mat_dict(zip_filename: str, data: np.ndarray, data_datetime: datetime) -> dict:
    mat_data = {'Data': data,
                'datetime': data_datetime.strftime('%Y/%m/%d-%HH'),
                'filename_origin': zip_filename
                }
    return mat_data


def save_mat_file(zip_filename: str, data: np.ndarray, data_datetime: datetime, output_path: str) -> None:
    mat_data = create_mat_dict(zip_filename, data, data_datetime)
    savemat(output_path, mat_data)


def build_path_output(data_datetime: datetime) -> tuple:
    year = data_datetime.year
    month = data_datetime.month
    day = data_datetime.day
    hour = data_datetime.hour

    output_folder = f'{output_root_folder}/mat_files/' + str(month).zfill(2) + str(day).zfill(2)
    output_path_file = output_folder + '/' + f'd_{str(year)[2:]}_{month}_{day}_{hour}.mat'
    return output_folder, output_path_file


def process_zip_files(week_list: Iterable[int], day_code_list: Iterable[str], hour_list: Iterable[int]):
    number_of_files_processed = 0

    for week in week_list:
        for day_code in day_code_list:
            for hour in hour_list:
                data_datetime = get_datetime(week, day_code, hour)
                output_folder, output_path_file = build_path_output(data_datetime)

                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder, exist_ok=True)

                if os.path.isfile(output_path_file):
                    continue

                zip_filename = f'{str(week).zfill(2)}{day_code}{str(hour).zfill(2)}.zip'
                zip_path = root_folder + '/' + zip_filename

                if os.path.isfile(zip_path):
                    data = process_zip(zip_path, zip_filename)

                    save_mat_file(zip_filename, data, data_datetime, output_path_file)

                    number_of_files_processed += 1
                    print(f"File {zip_path} processed successfully!")
                else:
                    print(f"File {zip_path} doesn't exist in the system")

    return number_of_files_processed


weeks = range(39, 45)
day_codes = day_map.keys()
hours = range(24)

files_processed_successfully = process_zip_files(weeks, day_codes, hours)
print(files_processed_successfully)
