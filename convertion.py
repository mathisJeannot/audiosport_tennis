import os
from moviepy.editor import *
from pydub import AudioSegment

def mp4_to_wav(path_mp4):
    path_mp3 = path_mp4[0:-3]+'mp3'
    path_wav = path_mp4[0:-3]+'wav'
    video = VideoFileClip(os.path.join(path_mp4))
    video.audio.write_audiofile(os.path.join(path_mp3))
    sound = AudioSegment.from_mp3(path_mp3)
    sound.export(path_wav, format="wav")
    
    
def frame_to_sec(nb_frames, fps):
    return nb_frames/fps

def sec_to_col(t, Fs, hop_length):
    return int(t*Fs/hop_length)