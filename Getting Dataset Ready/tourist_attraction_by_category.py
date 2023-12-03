import pandas as pd

file_path = 'C:/Users/ASUS/Downloads/Tourist Attraction_List.xlsx'
df = pd.read_excel(file_path)

result_df = pd.DataFrame(columns=['category', 'place_count', 'average_rating', 'total_reviews_count'])

grouped_df = df.groupby('category').agg({'place_id': 'count', 'rating': 'mean', 'reviews_count': 'sum'}).reset_index()

result_df['category'] = grouped_df['category']
result_df['place_count'] = grouped_df['place_id']
result_df['average_rating'] = grouped_df['rating']
result_df['total_reviews_count'] = grouped_df['reviews_count']

result_df.to_excel('C:/Users/ASUS/Downloads/Tourist Attraction_by_Category.xlsx', index=False)