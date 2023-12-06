# -*- coding: utf-8 -*-
"""User Rating_Filter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17T3-weAoXOT4b6tcTiz-W7KTHOqFI6fJ
"""

import pandas as pd

# Membaca file Excel
file_path = 'C:\Users\ASUS\Documents\Projects\Bangkit\Capstone\ML-Capstone-Project\Getting Dataset Ready\User Rating Tempat Wisata Merged.xlsx'
df = pd.read_excel(file_path)

# Menyimpan baris yang memiliki nilai tidak null di setidaknya tiga kolom
df_filtered = df.dropna(thresh=10)

# Menyimpan DataFrame yang telah difilter ke file Excel
df_filtered.to_excel('C:\Users\ASUS\Documents\Projects\Bangkit\Capstone\ML-Capstone-Project\Getting Dataset Ready\User Rating Tempat Wisata Merged_Filtered.xlsx', index=False)