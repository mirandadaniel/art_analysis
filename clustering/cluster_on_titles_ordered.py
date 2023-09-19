from PIL import Image
import numpy as np
import random
import re 

collage_width = 1000 
collage_height = 1000 
collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

text_file_path = '/art_class/clustering/x_art_names.txt' 
# specify the title name file that you want to cluster on 

strings = []
age_pattern = r'\b\d{4}\.jpg\b'

with open(text_file_path, 'r',  encoding='utf-8') as file:
    for line in file:
        split_values = line.strip().split(',')
        cleaned_strings = [value.strip(' "').strip("'") for value in split_values]
        strings.extend(cleaned_strings)

date_strs = []

for s in strings:
    match = re.search(age_pattern, s)
    if match:
        date_strs.append(s)
    else:
        continue

num_to_select = 100
str_subset = random.sample(date_strs, num_to_select)

date_dict = {}

for s in str_subset:
    match = re.search(age_pattern, s)
    if match:
        age_str = match.group()
        age_part = age_str[:-4]
        match_num = int(age_part)
        date_dict[s] = match_num
    else:
        continue

sorted_list = sorted(date_dict.items(), key = lambda x:x[1])

count =  0
for key, value in sorted_list:
    full_fp = '/art_class/style_no_dupes/' + key
    img = Image.open(full_fp)
    img = img.resize((collage_width // 10, collage_height // 10)) 
    x = (count % 10) * (collage_width // 10)
    y = (count // 10) * (collage_height // 10)
    collage.paste(img, (x, y))
    count+=1

collage.save('/art_class/clustering/x_collage_ordered.jpg')
    
