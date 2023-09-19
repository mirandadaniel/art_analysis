import os
import json

title_words = ["mountain", "party", "landscape", "dance", "portrait", "woman", "study", "young", "view", "child", "man", "old"]

artwork_dict = {'mountain': [],
                'party': [],
                'landscape': [],
                'dance': [],
                'portrait': [],
                'woman': [], 
                'study': [], 
                'young': [],
                'view': [],
                'child': [],
                'man': [],
                'old': []}

directory = os.fsencode("/art_class/style_no_dupes/")

for file in os.listdir(directory):
    artwork_filename = os.fsdecode(file)
    for title_word in artwork_dict:
        if title_word in artwork_filename:
            artwork_dict[title_word].append(artwork_filename)

resized_heatmap_arrays = [] 

heatmap_filenames = artwork_dict["woman"]

with open('/home/daniel/art_class/clustering/dance_art_names.txt', 'w') as convert_file:
     convert_file.write(json.dumps(heatmap_filenames))