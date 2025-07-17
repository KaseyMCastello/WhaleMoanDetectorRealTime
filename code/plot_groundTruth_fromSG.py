import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

# === Parameters ===
spectrogram_dir = r"C:\Users\Kasey\Desktop\TestMichaelaProgram\GroundTruthTests\GroundTruthSGs"
csv_path = r"E:\Lab Work\RealtimeWork\Ground_Truth_SOCAL34N_sitN_090722_200000.x.csv"
output_dir = os.path.join(spectrogram_dir, "with_gt_boxes")
os.makedirs(output_dir, exist_ok=True)

# === Spectrogram settings ===
window_size = 60       # seconds
time_per_pixel = 0.1   # seconds per horizontal pixel
freq_resolution = 1    # Hz per vertical pixel
max_freq = 150         # Hz

# === Helper: parse start time from wav_file_path ===
def get_xwav_start_time(wav_file_path):
    # Assumes path ends in something like: SOCAL34N_sitN_090722_200000.x.wav
    base = os.path.basename(wav_file_path)
    name = base.split('.')[0]  # SOCAL34N_sitN_090722_200000
    date_part = name.split('_')[-2:]  # ['090722', '200000']
    date_str = '20' + date_part[0] + '_' + date_part[1]  # '20090722_200000'
    return name, datetime.strptime(date_str, "%Y%m%d_%H%M%S")

# === Load CSV ===
df = pd.read_csv(csv_path)
df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')

# === Add image_file_path based on start_time ===
def assign_image_and_box(row):
    base_name, xwav_start = get_xwav_start_time(row['wav_file_path'])
    
    # Figure out which 60-second chunk this falls in
    chunk_start_sec = int(row['start_time'] // window_size * window_size)
    chunk_start_dt = xwav_start + timedelta(seconds=chunk_start_sec)
    timestamp = chunk_start_dt.strftime("%Y%m%dT%H%M%S")
    
    # Filename format: base.x_YYYYMMDDTHHMMSS.png
    image_filename = f"{base_name}.x_{timestamp}.png"
    image_path = os.path.join(spectrogram_dir, image_filename)
    
    # Pixel coordinates
    row['image_file_path'] = image_path
    row['chunk_start_sec'] = chunk_start_sec
    row['box_x1'] = (row['start_time'] - chunk_start_sec) / time_per_pixel
    row['box_x2'] = (row['end_time'] - chunk_start_sec) / time_per_pixel
    row['box_y1'] = (max_freq - row['high_f']) / freq_resolution
    row['box_y2'] = (max_freq - row['low_f']) / freq_resolution
    return row

df = df.apply(assign_image_and_box, axis=1)

# === Plot boxes per image ===
grouped = df.groupby('image_file_path')

for image_path, group in grouped:
    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_path}")
        continue

    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for _, row in group.iterrows():
        xmin, ymin = row['box_x1'], row['box_y1']
        xmax, ymax = row['box_x2'], row['box_y2']
        label = row['label']
        color = 'cyan' if label in ['A', 'B', 'D'] else 'pink'
        
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=3)
        draw.text((xmin, ymin - 12), label, fill=color, font=font)

    out_path = os.path.join(output_dir, os.path.basename(image_path))
    image.save(out_path)
    print(f"Saved with boxes: {out_path}")