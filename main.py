from datetime import datetime

from src.dataset.Z24Dataset import Z24Dataset

first_date = datetime(1998, 2, 1)
last_date = datetime(1998, 2, 2)
sensor_number = 0


dataset = Z24Dataset.load(first_date, last_date, sensor_number)
dataset.reshape_in_sequences(50, True)

print(dataset.data.shape)
