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
from Pylette import extract_colors
from PIL import Image
import numpy as np

warm_cool_rgb_cutoff = 48 # if <= cool, else warm
alg_switchpoint = 62 # if v of skin <= switch to modified contrast alg
alg_switchpoint2 = 50 # if v of skin < switch to alt mod alg
contrast_wh = 55 # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter
contrast_alt = 45 # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter
contrast_alt2 = 20 # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter

is_warm = 0 # if 0, cool else warm
is_high_contrast = 0 # if 0 low contrast, else high contrast

def is_warm_cool(skin_tone):
    r = skin_tone.rgb[0]
    b = skin_tone.rgb[2]
    rb_diff = r - b
    if (rb_diff <= warm_cool_rgb_cutoff):
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

    if (skin_v < alg_switchpoint2):
        if (max_diff < contrast_alt2):
            is_high_contrast = 0
            return is_high_contrast
        else:
            is_high_contrast = 1
            return is_high_contrast
    elif (skin_v <= alg_switchpoint):
        if (max_diff < contrast_alt):
            is_high_contrast = 0
            return is_high_contrast
        else:
            is_high_contrast = 1
            return is_high_contrast
    else:
        if (max_diff < contrast_wh):
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

    print(skin_tone.rgb)
    print(hair_color.rgb)
    print(eye_color.rgb)

    is_warm = is_warm_cool(skin_tone)
    is_high_contrast = is_high_low_contrast(skin_tone, hair_color, eye_color)

    result = [is_warm, is_high_contrast]
    return result

def parse_color_analysis_results(result):
    warm_cool = result[0]
    contrast = result[1]

    if (warm_cool == 0 and contrast == 0):
        return "summer"
    elif (warm_cool == 0 and contrast == 1):
        return "winter"
    elif (warm_cool == 1 and contrast == 1):
        return "autumn"
    else:
        return "spring"
    
def display_season_palette(season):
    if (season == "summer"):
        palette = extract_colors(image='color_analysis/season-palettes/cool-summer.JPG', palette_size=48)
    elif (season == "winter"):
        palette = extract_colors(image='color_analysis/season-palettes/cool-winter.JPG', palette_size=48)
    elif (season == "autumn"):
        palette = extract_colors(image='color_analysis/season-palettes/warm-autumn.JPG', palette_size=48)
    else:
        palette = extract_colors(image='color_analysis/season-palettes/warm-spring.JPG', palette_size=48)
    
    palette.display(save_to_file=True, filename="combined-demo/output-imgs/your-palette")
    return

def save_season_palette(season):
    if (season == "summer"):
        palette = extract_colors(image='color_analysis/season-palettes/cool-summer.JPG', palette_size=48)
    elif (season == "winter"):
        palette = extract_colors(image='color_analysis/season-palettes/cool-winter.JPG', palette_size=48)
    elif (season == "autumn"):
        palette = extract_colors(image='color_analysis/season-palettes/warm-autumn.JPG', palette_size=48)
    else:
        palette = extract_colors(image='color_analysis/season-palettes/warm-spring.JPG', palette_size=48)

    w, h = 48, 48
    img = Image.new("RGB", size=(w * palette.number_of_colors, h))
    arr = np.asarray(img).copy()
    for i in range(palette.number_of_colors):
        c = palette.colors[i]
        arr[:, i * h : (i + 1) * h, :] = c.rgb
    img = Image.fromarray(arr, "RGB")

    img.save(f"combined_demo/output-imgs/your-palette.jpg")
    
# python script code
# import argparser
import argparse

# process args
parser = argparse.ArgumentParser(description="Detecting and cropping face")
parser.add_argument("--input", "-i", help="Input image filename", dest="input", default="../example_images/00.JPG")
args = parser.parse_args()

# load input image
IMAGE_FILE = args.input

# run color analysis
color_analysis_result = color_analysis(IMAGE_FILE)

# separate color analysis results
season = parse_color_analysis_results(color_analysis_result)
print(f'your color season is {season}')

# display color palette
# display_season_palette(season)
save_season_palette(season)