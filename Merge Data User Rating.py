import os
import pandas as pd

folder_path = 'C:/Users/ASUS/Downloads/User Rating Tempat Wisata'

dfs = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        dfs.append(df)

from functools import reduce
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['id', 'username'], how='outer'), dfs)

merged_df =merged_df.fillna(0)

merged_df.to_excel('C:/Users/ASUS/Downloads/Book6.xlsx', index=False)