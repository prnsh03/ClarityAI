
# Preprocessing Videos for Multimodal RAG 


from pathlib import Path
import os
from os import path as osp
import json
import cv2
import webvtt
import whisper
from moviepy.editor import VideoFileClip
from PIL import Image
import base64


# Download Video Corpuses



from utils import download_video, get_transcript_vtt
# first video's url
vid1_url = "https://www.youtube.com/watch?v=15PK38MUEPM"

# download Youtube video to ./shared_data/videos/video1
vid1_dir = "./shared_data/videos/video1"
vid1_filepath = download_video(vid1_url, vid1_dir)

# download Youtube video's subtitle to ./shared_data/videos/video1
vid1_transcript_filepath = get_transcript_vtt(vid1_url, vid1_dir)



# show the paths to video1 and its transcription
print(vid1_filepath)
print(vid1_transcript_filepath)


get_ipython().system('head -n15 {vid1_transcript_filepath}')


from urllib.request import urlretrieve
# second video's url
vid2_url="https://www.youtube.com/watch?v=nVyD6THcvDQ"
vid2_dir = "./shared_data/videos/video2"
vid2_name = "AI.mp4"

# create folder to which video2 will be downloaded 
Path(vid2_dir).mkdir(parents=True, exist_ok=True)
vid2_filepath = urlretrieve(
                        vid2_url, 
                        osp.join(vid2_dir, vid2_name)
                    )[0]

from utils import str2time
from utils import maintain_aspect_ratio_resize


# ## 1. Video Corpus and Its Transcript Are Available


# function `extract_and_save_frames_and_metadata``:
#   receives as input a video and its transcript
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata(
        path_to_video, 
        path_to_transcript, 
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    # load transcript using webvtt
    trans = webvtt.read(path_to_transcript)
    
    # iterate transcript file
    # for each video segment specified in the transcript file
    for idx, transcript in enumerate(trans):
        # get the start time and end time in seconds
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        # get the time in ms exactly 
        # in the middle of start time and end time
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        # get the transcript, remove the next-line symbol
        text = transcript.text.replace("\n", ' ')
        # get frame at the middle time
        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()
        if success:
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                path_to_save_extracted_frames, img_fname
            )
            cv2.imwrite(img_fpath, image)

            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    # save metadata of all extracted frames
    fn = osp.join(path_to_save_metadatas, 'metadatas.json')
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas


# output paths to save extracted frames and their metadata 
extracted_frames_path = osp.join(vid1_dir, 'extracted_frame')
metadatas_path = vid1_dir

# create these output folders if not existing
Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

# call the function to extract frames and metadatas
metadatas = extract_and_save_frames_and_metadata(
                vid1_filepath, 
                vid1_transcript_filepath,
                extracted_frames_path,
                metadatas_path,
            )
                     

metadatas[:4]


### 2. Video Corpus without Available Transcript


path_to_video_no_transcript = vid1_filepath

# declare where to save .mp3 audio
path_to_extracted_audio_file = os.path.join(vid1_dir, 'audio.mp3')

# extract mp3 audio file from mp4 video video file
clip = VideoFileClip(path_to_video_no_transcript)
clip.audio.write_audiofile(path_to_extracted_audio_file)

model = whisper.load_model("small")
options = dict(task="translate", best_of=1, language='en')
results = model.transcribe(path_to_extracted_audio_file, **options)


from utils import getSubs
vtt = getSubs(results["segments"], "vtt")

# path to save generated transcript of video1
path_to_generated_trans = osp.join(vid1_dir, 'generated_video1.vtt')
# write transcription to file
with open(path_to_generated_trans, 'w') as f:
    f.write(vtt)



get_ipython().system('head {path_to_generated_trans}')


# ## 3. Video Corpus without Language


lvlm_prompt = "Can you describe the image?"


# ### LVLM Inference Example

path_to_frame = osp.join(vid1_dir, "extracted_frame", "frame_5.jpg")
frame = Image.open(path_to_frame)
frame

from utils import lvlm_inference, encode_image
# need to encode this frame with base64 encoding 
#  as input image to function lvlm_inference
# encode image to base64
image = encode_image(path_to_frame)
caption = lvlm_inference(lvlm_prompt, image)
print(caption)


# Extract Frames and Metadata for Videos Using LVLM Inference


# function extract_and_save_frames_and_metadata_with_fps
#   receives as input a video 
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata_with_fps(
        path_to_video,  
        path_to_save_extracted_frames,
        path_to_save_metadatas,
        num_of_extracted_frames_per_second=1):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    
    # Get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # Get hop = the number of frames pass before a frame is extracted
    hop = round(fps / num_of_extracted_frames_per_second) 
    curr_frame = 0
    idx = -1
    while(True):
        # iterate all frames
        ret, frame = video.read()
        if not ret: 
            break
        if curr_frame % hop == 0:
            idx = idx + 1
        
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                            path_to_save_extracted_frames, 
                            img_fname
                        )
            cv2.imwrite(img_fpath, image)

            # generate caption using lvlm_inference
            b64_image = encode_image(img_fpath)
            caption = lvlm_inference(lvlm_prompt, b64_image)
                
            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)
        curr_frame += 1
        
    # save metadata of all extracted frames
    metadatas_path = osp.join(path_to_save_metadatas,'metadatas.json')
    with open(metadatas_path, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas


# paths to save extracted frames and metadata (their transcripts)
extracted_frames_path = osp.join(vid2_dir, 'extracted_frame')
metadatas_path = vid2_dir

# create these output folders if not existing
Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

# call the function to extract frames and metadatas
metadatas = extract_and_save_frames_and_metadata_with_fps(
                vid2_filepath, 
                extracted_frames_path,
                metadatas_path,
                num_of_extracted_frames_per_second=0.1
            )


data = metadatas[1]
caption = data['transcript']
print(f'Generated caption is: "{caption}"')
frame = Image.open(data['extracted_frame_path'])
display(frame)


