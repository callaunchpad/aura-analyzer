""" 
init vars for color analysis sorting 

warm_cool_rgb_cutoff calculated via random sample of people from each of the 
12 color seasons using rgb values of skin tone -- given the fact that all human skin
tones will have a baseline amount of redness, we found that the difference between red value
and blue value will always be positive. however, higher blue values or cool toned skin
will result in a smaller difference between the two while higher red values or warm toned
skin will result in a bigger difference between the two. we assigned a value of 88 to be
the cutoff r-b difference, where differences less than or equal to 88 indicates cool tones
and differences more than 88 indicates warm tones

contrast vars:
we switch to hsv values to better calculate contrast
contrast_cutoff_wh calcuated via random sample of people from each of the 
12 color seasons using hsv values of skin tone and eyes. 

contrast_wh

alg_switchpoint is rough estimate of when a person might have darker skin and contrast will be lower.
we will take this value by finding the hsv of skin tone and taking the v value


"""

import sqlite3
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd
from Pylette import extract_colors
from tqdm import tqdm

warm_cool_rgb_cutoff = 48  # if <= cool, else warm
alg_switchpoint = 62  # if v of skin <= switch to modified contrast alg
alg_switchpoint2 = 50  # if v of skin < switch to alt mod alg
contrast_wh = 55  # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter
contrast_alt = 45  # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter
contrast_alt2 = 20  # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter

is_warm = 0  # if 0, cool else warm
is_high_contrast = 0  # if 0 low contrast, else high contrast

con = sqlite3.connect("small-fashion-dataset.db")


def setup_fashion_table():
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

    for file in tqdm(files):
        # Get season
        color_analysis_result = color_analysis(file)
        season = parse_color_analysis_results(color_analysis_result)

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
    cur.execute(
        """INSERT INTO fashion(id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName, colorSeason)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        col_vals,
    )
    con.commit()


def is_warm_cool(skin_tone):
    r = skin_tone.rgb[0]
    b = skin_tone.rgb[2]
    rb_diff = r - b
    if rb_diff <= warm_cool_rgb_cutoff:
        is_warm = 0
        return is_warm
    else:
        is_warm = 1
        return is_warm


def is_high_low_contrast(skin_tone, hair_color, eye_color):
    skin_tone = skin_tone.hsv
    hair_color = hair_color.hsv
    eye_color = eye_color.hsv

    skin_v = skin_tone[2]
    hair_v = hair_color[2]
    eye_v = eye_color[2]

    hair_diff = abs(skin_v - hair_v)
    eye_diff = abs(skin_v - eye_v)
    max_diff = max(hair_diff, eye_diff)

    if skin_v < alg_switchpoint2:
        if max_diff < contrast_alt2:
            is_high_contrast = 0
            return is_high_contrast
        else:
            is_high_contrast = 1
            return is_high_contrast
    elif skin_v <= alg_switchpoint:
        if max_diff < contrast_alt:
            is_high_contrast = 0
            return is_high_contrast
        else:
            is_high_contrast = 1
            return is_high_contrast
    else:
        if max_diff < contrast_wh:
            is_high_contrast = 0
            return is_high_contrast
        else:
            is_high_contrast = 1
            return is_high_contrast


def color_analysis(image_path):
    palette = extract_colors(image=image_path, palette_size=3, sort_mode="luminance")

    # palette.display()
    skin_tone = palette.colors[2]
    hair_color = palette.colors[1]
    eye_color = palette.colors[0]

    # print(skin_tone.rgb)
    # print(hair_color.rgb)
    # print(eye_color.rgb)

    is_warm = is_warm_cool(skin_tone)
    is_high_contrast = is_high_low_contrast(skin_tone, hair_color, eye_color)

    result = [is_warm, is_high_contrast]
    return result


def parse_color_analysis_results(result):
    warm_cool = result[0]
    contrast = result[1]

    if warm_cool == 0 and contrast == 0:
        return "summer"
    elif warm_cool == 0 and contrast == 1:
        return "winter"
    elif warm_cool == 1 and contrast == 1:
        return "autumn"
    else:
        return "spring"


def display_season_palette(season):
    if season == "summer":
        palette = extract_colors(image="../../color-analysis/season-palettes/cool-summer.JPG", palette_size=48)
    elif season == "winter":
        palette = extract_colors(image="../../color-analysis/season-palettes/cool-winter.JPG", palette_size=48)
    elif season == "autumn":
        palette = extract_colors(image="../../color-analysis/season-palettes/warm-autumn.JPG", palette_size=48)
    else:
        palette = extract_colors(image="../../color-analysis/season-palettes/warm-spring.JPG", palette_size=48)

    palette.display(save_to_file=True, filename="../output-imgs/your-palette")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detecting and cropping face")
    parser.add_argument("--input_dir", "-d", help="Input image file dir", dest="input_dir")
    parser.add_argument("--styles_dir", "-s", help="Style table dir", dest="styles_dir")
    args = parser.parse_args()

    dir = args.input_dir
    styles_dir = args.styles_dir

    setup_fashion_table()

    run_color_analysis(dir, styles_dir)
