import numpy as np
import os
from dotenv import load_dotenv

load_dotenv('.env')

print(os.environ['WORK_FOLDER'])
print(np.__version__)
