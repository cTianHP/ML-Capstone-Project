# -*- coding: utf-8 -*-
"""Transform Data User Rating.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DdphY4J5ibuAkve8P-Zel59_Pg6RgqQI
"""

import pandas as pd

df = pd.read_excel('User Rating Tempat Wisata Merged_Filtered.xlsx')

transform_df = pd.melt(df, id_vars=['user_id'], var_name='place_id', value_name='rating')
transform_df = transform_df.dropna(subset=['rating'])

sorted_df = transform_df.sort_values(by=['user_id', 'place_id'])


sorted_df.to_excel('Transformed User Rating Tempat Wisata.xlsx', index=False)