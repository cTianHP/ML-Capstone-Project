import pandas as pd

file1 = 'Rute 1.xlsx'
file2 = 'Rute 2.xlsx'
file3 = 'Rute 3 & 4.xlsx'

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)
df3 = pd.read_excel(file3)

merged_df = pd.concat([df1, df2, df3], ignore_index=True)

merged_df.to_excel('bus_route_merged.xlsx', index=False)