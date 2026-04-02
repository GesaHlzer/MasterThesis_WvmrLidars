# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 01:35:20 2025

@author: alleh
"""
# pip install pillow
import os
from PIL import Image

#%%

# Bilder laden
image1 = Image.open("bild1.png")
image2 = Image.open("bild2.png")

# Größe der Bilder (wir gehen davon aus, dass sie gleich groß sind)
width, height = image1.size

# Neues Bild mit doppelter Höhe erstellen
combined_image = Image.new("RGBA", (width, height * 2))

# Erstes Bild oben einfügen
combined_image.paste(image1, (0, 0))

# Zweites Bild unten einfügen
combined_image.paste(image2, (0, height))

# Neues Bild speichern
combined_image.save("kombiniert.png")



#%% Mehrere Bilder untereinander (z. B. 3 Bilder)

# Liste der Bildpfade
image_paths = ["bild1.png", "bild2.png", "bild3.png"]

# Bilder laden
images = [Image.open(path) for path in image_paths]

# Wir gehen davon aus, dass alle Bilder gleich groß sind
width, height = images[0].size

# Neues Bild mit entsprechender Höhe erstellen
combined_height = height * len(images)
combined_image = Image.new("RGBA", (width, combined_height))

# Bilder nacheinander einfügen
for i, img in enumerate(images):
    combined_image.paste(img, (0, i * height))

# Speichern
combined_image.save("kombiniert_vertikal.png")



#%% 2×2 Grid (z. B. 4 Bilder)

# Bildpfade
image_paths = ["bild1.png", "bild2.png", "bild3.png", "bild4.png"]

# Bilder laden
images = [Image.open(path) for path in image_paths]

# Größe der Bilder
width, height = images[0].size

# Neues Bild mit 2×2 Grid erstellen
combined_image = Image.new("RGBA", (width * 2, height * 2))

# Bilder einfügen
combined_image.paste(images[0], (0, 0))               # oben links
combined_image.paste(images[1], (width, 0))           # oben rechts
combined_image.paste(images[2], (0, height))          # unten links
combined_image.paste(images[3], (width, height))      # unten rechts

# Speichern
combined_image.save("kombiniert_grid.png")

#%% 1x2 Grid

# 1. Pfade definieren
# input_folder = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots\Temperature_PPL20m-AWS_Raso"
# output_folder = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots\Temperature_PPL20m-AWS_Raso\compare_filtering"
input_folder = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots\WVMR_DA10-PPL20m-AWS_Raso"
output_folder = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots\WVMR_DA10-PPL20m-AWS_Raso\compare_filtering"

# 2. Zielordner anlegen, falls nicht vorhanden
os.makedirs(output_folder, exist_ok=True)

# 3. Alle PNG-Dateien sortiert einlesen
# png_files = sorted([
#     f for f in os.listdir(input_folder)
#     if f.lower().endswith('.png')
# ])
png_files = sorted([
    f for f in os.listdir(input_folder)
    if '12km' in f.lower()
    and f.lower().endswith('.png')
    ])
# png_files = png_files[:-1]
# 4. Bilderpaare kombinieren und speichern
combined_images = []
for i in range(0, len(png_files), 2):
    # Nur, wenn es noch ein zweites Bild gibt
    if i + 1 < len(png_files):
        path1 = os.path.join(input_folder, png_files[i])
        path2 = os.path.join(input_folder, png_files[i + 1])

        img1 = Image.open(path1)
        img2 = Image.open(path2)

        # Neue Bildgröße (Breite = Summe, Höhe = max)
        combined_width = img1.width + img2.width
        combined_height = max(img1.height, img2.height)
        combined_img = Image.new('RGB', (combined_width, combined_height))

        # Bilder nebeneinander einfügen
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))

        # Speichern
        out_name = f"combined_{i//2 + 1:03d}.png"
        out_path = os.path.join(output_folder, out_name)
        combined_img.save(out_path)

        combined_images.append(combined_img)

# 5. Als GIF speichern (1 Sekunde pro Frame, Endlosschleife)
if combined_images:
    gif_path = os.path.join(output_folder, "comparison.gif")
    combined_images[0].save(
        gif_path,
        save_all=True,
        append_images=combined_images[1:],
        duration=1000, # in ns
        loop=0
    )

print("done with GIF")

#%% Make Gif

folder = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\VerticalPlots\WVMR_DA10-PPL20m-AWS_Raso"

png_files = sorted([f for f in os.listdir(folder)
            if '3km' in f and 'filtered' in f and f.lower().endswith('.png')])
# png_files = png_files[:-1]
images = [Image.open(os.path.join(folder, png)) for png in png_files]

# folder_paths = r"C:\Users\alleh\Documents\+Uni_Innsbruck\+MasterThesis\plots\Lidar_Comparison\diff_timeseries_dt10s_rl1"
# image_paths = [file for file in os.listdir(folder_paths) if file.endswith(".png")]

gif_path = os.path.join(os.path.dirname(folder), "vertical_wvmr_to3km.gif")

images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=1000, # in ns
    loop=0
)
