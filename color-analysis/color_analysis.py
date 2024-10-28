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

warm_cool_rgb_cutoff = 88 # if <= cool, else warm
alg_switchpoint = 62 # if v of skin <= switch to modified contrast alg
alg_switchpoint2 = 50 # if v of skin < switch to alt mod alg
contrast_wh = 55 # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter
contrast_alt = 45 # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter
contrast_alt2 = 20 # if max(abs(skin_v - hair_v), abs(skin_v - eye_v)) <= spring/summer else autumn/winter

is_warm = 0 # if 0, cool else warm
is_high_contrast = 0 # if 0 low contrast, else high contrast

def is_warm_cool(skin_tone):
    r = skin_tone[0]
    b = skin_tone[2]
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
    palette = extract_colors(image=image_path, palette_size=3)
    skin_tone = palette[0]
    hair_color = palette[1]
    eye_color = palette[2]

    is_warm = is_warm_cool(skin_tone)
    is_high_contrast = is_high_low_contrast(skin_tone, hair_color, eye_color)

    result = [is_warm, is_high_contrast]
    return result



