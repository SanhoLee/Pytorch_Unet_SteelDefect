##
import os
from zipfile import ZipFile
import pandas as pd

data_dir = "./datasets"
zip_dir = os.path.join(data_dir, 'origin')

## read csv file and make dataframe.
df = pd.read_csv(os.path.join(data_dir,'train.csv'))


# organizing data-set
