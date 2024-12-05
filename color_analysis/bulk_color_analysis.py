""" 
possible base colors:
['Navy Blue' 'Blue' 'Silver' 'Black' 'Grey' 'Green' 'Purple' 'White'
 'Beige' 'Brown' 'Bronze' 'Teal' 'Copper' 'Pink' 'Off White' 'Maroon'
 'Red' 'Khaki' 'Orange' 'Coffee Brown' 'Yellow' 'Charcoal' 'Gold' 'Steel'
 'Tan' 'Multi' 'Magenta' 'Lavender' 'Sea Green' 'Cream' 'Peach' 'Olive'
 'Skin' 'Burgundy' 'Grey Melange' 'Rust' 'Rose' 'Lime Green' 'Mauve'
 'Turquoise Blue' 'Metallic' 'Mustard' 'Taupe' 'Nude' 'Mushroom Brown' nan
 'Fluorescent Green']

tldr map the base color in image to it's nearest neighbor in the color palettes, 
check which color palette that is and sort it into that season

"""

import sqlite3
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd
from Pylette import extract_colors
from tqdm import tqdm

import numpy as np
from typing import Tuple

import argparse

def setup_fashion_table(con):
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS fashion(
            id INTEGER PRIMARY KEY, 
            gender TEXT, 
            masterCategory TEXT, 
            subCategory TEXT, 
            articleType TEXT, 
            baseColour TEXT, 
            season TEXT, 
            year INTEGER, 
            usage TEXT, 
            productDisplayName TEXT,
            colorSeason TEXT
        )"""
    )


def run_color_analysis(dir: str, styles_dir):
    # retrieve the styles_dir
    style_table = pd.read_csv(styles_dir, index_col=0, on_bad_lines="warn")
    # get all files in dir
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

    # create color palettes for each season first
    summer_palette = extract_colors(image="nonspecific-season-palettes/summer-palette.jpg", palette_size=144, sort_mode="luminance")
    spring_palette = extract_colors(image="nonspecific-season-palettes/spring-palette.jpg", palette_size=144, sort_mode="luminance")
    winter_palette = extract_colors(image="nonspecific-season-palettes/winter-palette.jpg", palette_size=144, sort_mode="luminance")
    autumn_palette = extract_colors(image="nonspecific-season-palettes/autumn-palette.jpg", palette_size=144, sort_mode="luminance")

    for file in tqdm(files):
        # Get season
        season = color_analysis(file, spring_palette, summer_palette, winter_palette, autumn_palette)

        # Get the file name without extension, and excluding the path
        pid = int(Path(file).stem)
        try:
            addn_info_col_vals = retr_addn_info(pid, style_table)
        except KeyError:
            # if pid not in table, ignore
            continue

        # full values
        col_vals = addn_info_col_vals + (season,)

        # insert into db
        add_new_item_to_db(col_vals)


def retr_addn_info(pid: str, style_table):
    row = style_table.loc[pid]

    col_vals = (
        pid,
        row["gender"],
        row["masterCategory"],
        row["subCategory"],
        row["articleType"],
        row["baseColour"],
        row["season"],
        row["year"],
        row["usage"],
        row["productDisplayName"],
    )
    return col_vals

 
def add_new_item_to_db(col_vals):
    cur = con.cursor()
    # change to upsert
    cur.execute(
        """
        INSERT INTO fashion(id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName, colorSeason)
        VALUES(?,?,?,?,?,?,?,?,?,?,?) 
        ON CONFLICT(id) DO UPDATE SET
            gender=excluded.gender,
            masterCategory=excluded.masterCategory,
            subCategory=excluded.subCategory,
            articleType=excluded.articleType,
            baseColour=excluded.baseColour,
            season=excluded.season,
            year=excluded.year,
            usage=excluded.usage,
            productDisplayName=excluded.productDisplayName,
            colorSeason=excluded.colorSeason
        """,
        col_vals,
    )
    con.commit()
    

def color_analysis(image_path, spring, summer, winter, autumn):
    palette = extract_colors(image=image_path, palette_size=2, sort_mode="frequency")
    most_freq_color = palette.colors[0]
    
    # check for white background
    if ((most_freq_color.rgb == np.array([255, 255, 255])).all()):
        most_freq_color = palette.colors[1]
    
    palette_distances = {}
    palette_distances['spring'] = find_closest_color(most_freq_color, spring)
    palette_distances['summer'] = find_closest_color(most_freq_color, summer)
    palette_distances['winter'] = find_closest_color(most_freq_color, winter)
    palette_distances['autumn'] = find_closest_color(most_freq_color, autumn)

    if (palette_distances['spring'] == min(palette_distances.values())):
        return 'spring'
    elif (palette_distances['summer'] == min(palette_distances.values())):
        return 'summer'
    elif (palette_distances['winter'] == min(palette_distances.values())):
        return 'winter'
    else:
        return 'autumn'

def find_closest_color(target_color, target_palette):
    """
    Finds the closest color to the target color for a target palette

    Parameters:
        target_color (Color): The color to compare against.
        target_palettes (Palette): The color palette to search

    Returns:
        Tuple[Color, Palette]: The closest color and the palette it belongs to.
    """
    def color_distance(c1, c2) -> float:
        """
        Calculates the Euclidean distance between two colors in RGB space.
        
        Parameters:
            c1 (Color): The first color.
            c2 (Color): The second color.

        Returns:
            float: The Euclidean distance between c1 and c2.
        """
        return np.linalg.norm(np.array(c1.rgb) - np.array(c2.rgb))

    closest_color = None
    min_distance = float('inf')

    for color in target_palette.colors:
        distance = color_distance(target_color, color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color

    return min_distance


def display_season_palette(season):
    if season == "summer":
        palette = extract_colors(image="../../color_analysis/season-palettes/cool-summer.JPG", palette_size=48)
    elif season == "winter":
        palette = extract_colors(image="../../color_analysis/season-palettes/cool-winter.JPG", palette_size=48)
    elif season == "autumn":
        palette = extract_colors(image="../../color_analysis/season-palettes/warm-autumn.JPG", palette_size=48)
    else:
        palette = extract_colors(image="../../color_analysis/season-palettes/warm-spring.JPG", palette_size=48)

    palette.display(save_to_file=True, filename="../output-imgs/your-palette")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Color analyze clothing items")
    parser.add_argument("--input_dir", "-d", help="Input image file dir", dest="input_dir")
    parser.add_argument("--styles_dir", "-s", help="Style table dir", dest="styles_dir")
    parser.add_argument("--db_name", "-db", help="Name of database", dest="db_name")
    args = parser.parse_args()

    dir = args.input_dir
    styles_dir = args.styles_dir
    db_name = args.db_name
    
    # con = sqlite3.connect("small-fashion-dataset.db")
    con = sqlite3.connect(db_name)

    setup_fashion_table(con)

    run_color_analysis(dir, styles_dir)
