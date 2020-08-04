
import cv2 as cv
import argparse
from pathlib import Path

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('path_to_video', type=str, nargs=1,
                    help='Path to the video you want to split')
parser.add_argument('--every', type=int, nargs=1,
                    help='Save every ith frame')

args = parser.parse_args()
video_path = Path(args.path_to_video[0])
video_name = video_path.stem

if args.every:
    every_ith_frame = args.every[0]

    if every_ith_frame < 1:
        print('Usage Error: --every takes an integer > 0')
        exit(1)

else:
    every_ith_frame = 'None'

# Setup
output_path = Path.cwd()
output_path = output_path / video_name
try:
    output_path.mkdir()
except FileExistsError:
    print('Error: Directory ' + video_name + ' already exists in current directory')
    exit(1)
print('Created a new directory called ' + str(video_name) + ' in the current directory')

# Instructions
if every_ith_frame is 'None':
    print('''You are manually choosing the frames to save.
    Press q to exit
    Press s to save a frame
    Press any other key to see next frame
    
If you would like to save every ith frame instead run:
python3 video_2_frames --every <integer> <path_to_video>
''')

# Video Processing
cap = cv.VideoCapture(str(video_path))

i = 0
while True:

    i += 1

    ret_val, next_frame = cap.read() # Reads the next video frame into memory

    if ret_val is False:
        break

    if every_ith_frame is not 'None':
        if i % every_ith_frame is 0:
            image_name = 'frame_' + str(i) + '.jpg'
            cv.imwrite(str(output_path / image_name), next_frame)
        continue

    cv.imshow('frame'+str(i),next_frame)

    key = cv.waitKey(0)
    if key == 113: # Hit q key to exit
        break
    elif key == 115: # Hit s key to save
        image_name = 'frame_' + str(i) + '.jpg'
        cv.imwrite(str(output_path / image_name), next_frame)
    else:
        pass

    cv.destroyAllWindows()
cap.release()


