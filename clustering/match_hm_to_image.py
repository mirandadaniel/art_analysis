import cv2
import os

heatmap_dir = os.fsencode('/art_class/clustering/cosine_clusters/heatmaps')  
heatmap_dir_str = '/art_class/clustering/cosine_clusters/heatmaps'
art_dir = os.fsencode('/art_class/wikiart/') 
art_dir_str = '/art_class/wikiart/'
output_directory = '/art_class/clustering/cosine_clusters/complete_images/'
match_found = False
heatmap_filenames = [os.fsdecode(file) for file in os.listdir(heatmap_dir)]

for heatmap_filename in heatmap_filenames:
    for subdir, dirs, files in os.walk(art_dir_str):
        subdir_name = os.path.basename(subdir)
        for file in files:
            artwork_filename = os.path.basename(file) 
            full_artwork_path = os.path.join(subdir_name, file)
            full_artwork_path = os.path.join(art_dir_str, full_artwork_path)
            if artwork_filename in heatmap_filename:
                overlay_filename = "/art_class/clustering/cosine_clusters/top_heatmaps/" + heatmap_filename
                background_filename = full_artwork_path
                overlay = cv2.imread(overlay_filename)
                background = cv2.imread(background_filename)
                overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
                added_image = cv2.addWeighted(background, 0.6, overlay, 0.4, 0)
                final_fp = output_directory + heatmap_filename
                cv2.imwrite(final_fp, added_image)