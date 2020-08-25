#!/bin/sh

# sh sample_videos.sh path_to_videos <every_ith_frame> <framerate>

# Given a folder of videos this script will sample every ith frame
# and create a new videos at a chosen framerate.

# Need ffmpeg and video_2_frames.py script

mkdir temp_frames || { echo 'Directory temp_frames exists, remove to run this program' ; exit 1; }
mkdir output_videos || { echo 'Directory output_videos exists, remove to run this program' ; rm -r temp_frames; exit 1; }

path_to_vids="$1"
ith="$2"
framerate="$3"

vid_list=$(ls "$path_to_vids")

num_videos=$(ls "$path_to_vids" | wc -l)

count=1
for f in $vid_list ;do
    echo "Video number: $count/$num_videos"
    file_path="$path_to_vids/$f"
    folder_name=$(echo "$f" | sed 's/\.[a-zA-Z]*$//')
    python3 video_2_frames.py --every "$ith" --output_path "$(pwd)/temp_frames" "$file_path"
    ffmpeg -framerate "$framerate" -i "temp_frames/$folder_name/frame_%d.jpg" "output_videos/$folder_name.mp4"
    count=$((count + 1))
done

rm -r temp_frames