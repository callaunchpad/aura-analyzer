import os
import pandas as pd

from grailed_api import GrailedAPIClient
from grailed_api.enums import Conditions, Markets, Locations, Departments
import json

import requests
import argparse

def download_image(save_path, image_url):
    with open(save_path, 'wb') as handle:
        response = requests.get(image_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

def get_products(client, product_specifications):

    products = client.find_products(
        conditions=[Conditions.IS_GENTLY_USED, Conditions.IS_NEW],
        markets=[Markets.BASIC, Markets.GRAILED],
        locations=[Locations.US, Locations.ASIA, Locations.EUROPE],
        **product_specifications
    )
    return products

def fix_category(row):
    # gender
    if row["gender"]=="menswear":
        row["gender"] = "Men"
    else:
        row["gender"] = "Women"
    # category
    if row["masterCategory"]=="tops" or row["masterCategory"]=="outerwear":
        row["subCategory"] = "Topwear"
        row["masterCategory"] = "Apparel"
    elif row["masterCategory"]=="bottoms":
        row["subCategory"] = "Bottomwear"
        row["masterCategory"] = "Apparel"
    elif row["masterCategory"]=="accessories":
        row["masterCategory"]=="Accessories"
    elif row["masterCategory"]=="footwear":
        row["subCategory"]=="footwear"
        row["masterCategory"]=="Accessories"

    return row

def download_and_save_products(products):
    data = []
    for product in products:
        product_id = product["cover_photo"]["listing_id"]
        image_url = product["cover_photo"]["image_url"]
        save_path = f"{data_path}/images/{product_id}.jpg"
        data.append(
            {
                "id": product_id,
                "gender": product["department"],
                "masterCategory": product["category"],
                "subCategory": product["category_path"],
                "articleType": product["category_path_size"],
                "baseColour": product["color"],
                "season": None,
                "year": None,
                "usage": "Casual", 
                "productDisplayName": product["description"]
            }
        )
        # check if photo already downloaded
        if not os.path.exists(save_path):
            download_image(save_path, image_url)

    # create empty dataframe
    df = pd.DataFrame(data=data, columns=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName'])
    df = df.apply(fix_category, axis=1)
    print(df.head())

    df.to_csv(f'{data_path}/styles_grailed.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetching and downloading items from Grailed")
    parser.add_argument("--gender", "-g", help="Gender, mens or womens", dest="department")
    parser.add_argument("--query_search", "-q", help="Query to look up", dest="query_search")
    parser.add_argument("--num_hits", "-n", help="Number of hits to pull up", dest="hits_per_page", default=1)
    # parser.add_argument("--categories", "-c", helper="Iterable of categories to include")
    parser.add_argument("--price_from", "-l", help="Min price", dest="price_from", default=0)
    parser.add_argument("--price_to", "-m", help="Max price", dest="price_to", default=100)
    parser.add_argument("--data_path", "-d", help="Directory to store items", dest="data_path", default="grailed-dataset")
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(f"{data_path}/images"):
        os.makedirs(f"{data_path}/images")

    # start client
    client = GrailedAPIClient()

    # fetch results
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_dict.pop("data_path")
    print(args_dict)
    products = get_products(client, args_dict)

    # download images, save products as csv
    download_and_save_products(products)