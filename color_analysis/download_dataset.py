import os
import pandas as pd

# df = pd.read_csv(f'grailed-dataset/styles_grailed.csv')
# print(df.head())
# print(len(df.loc[df["Department"]=="womenswear"]))
num_hits = 40
page_range = 500
for department in ["menswear", "womenswear"]:
    for page_num in range(page_range):
        os.system(f"python3 get_grailed_items.py -g {department} -n {num_hits} -p {page_num}") 