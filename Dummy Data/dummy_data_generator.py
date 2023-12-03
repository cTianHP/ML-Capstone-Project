import pandas as pd
import numpy as np

jumlah_baris = 2
jumlah_kolom = 3

df = pd.DataFrame()

df['user_id'] = np.arange(1, jumlah_baris + 1)

for i in range(1, jumlah_kolom):
    nama_kolom = f'tempat_{i}'
    df[nama_kolom] = np.random.randint(0, 6, jumlah_baris)

df.replace(0, np.nan, inplace=True)

df.to_excel('dummy_data.xlsx', index=False)