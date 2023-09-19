from PIL import Image
import numpy as np
import random
import cv2
import os

collage_width = 1000
collage_height = 1000
collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

text_file_path = '/art_class/clustering/dance_art_names.txt'

strings = []

with open(text_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        split_values = line.strip().split(',')
        cleaned_strings = [value.strip(' "').strip("'") for value in split_values]
        strings.extend(cleaned_strings)

num_to_select = 100
str_subset = random.sample(strings, num_to_select)

count = 0
for s in str_subset:
    full_fp = '/art_class/style_no_dupes/' + s
    img = Image.open(full_fp)
    img = img.resize((collage_width // 10, collage_height // 10))

    x = (count % 10) * (collage_width // 10)
    y = (count // 10) * (collage_height // 10)

    collage.paste(img, (x, y))

    count += 1

overlay_images = []

heatmap_dir_str = '/art_class/clustering/cosine_clusters/top_heatmaps'
art_dir_str = '/art_class/wikiart/'

heatmap_filenames = os.listdir(heatmap_dir_str)

for heatmap_filename in heatmap_filenames:
    for subdir, dirs, files in os.walk(art_dir_str):
        subdir_name = os.path.basename(subdir)
        for file in files:
            artwork_filename = os.path.basename(file)
            full_artwork_path = os.path.join(subdir_name, file)
            full_artwork_path = os.path.join(art_dir_str, full_artwork_path)
            if artwork_filename in heatmap_filename:
                overlay_images.append((full_artwork_path, heatmap_filename))
                break

output_directory = '/art_class/clustering/cosine_clusters/complete_images/'
os.makedirs(output_directory, exist_ok=True)

for background_filename, heatmap_filename in overlay_images:
    overlay_filename = os.path.join(heatmap_dir_str, heatmap_filename)
    overlay = cv2.imread(overlay_filename)
    background = cv2.imread(background_filename)

    overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
    added_image = cv2.addWeighted(background, 0.6, overlay, 0.4, 0)
    final_fp = os.path.join(output_directory, heatmap_filename)
    cv2.imwrite(final_fp, added_image)

    added_img_pil = Image.open(final_fp)
    added_img_pil = added_img_pil.resize((collage_width // 10, collage_height // 10))
    added_img_pil.save(final_fp)

    img = Image.open(final_fp)
    collage.paste(img, (x, y))

collage.save('/art_class/clustering/dance_collage.jpg')