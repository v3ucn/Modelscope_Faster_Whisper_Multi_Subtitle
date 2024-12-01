import argparse
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ffmpeg_path = f"{ROOT_DIR}/bin" # 替换成你的 FFmpeg bin 目录
# os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffmpeg_path

import gradio as gr

from utils import movie2audio,make_srt,make_tran,merge_sub,make_tran_zh2en,make_tran_ja2zh,make_tran_ko2zh,make_srt_sv,make_tran_qwen2,make_tran_deep

from subtitle_to_audio import generate_audio
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')       # getting details of current voice
vlist = []
num = 0
for voice in voices:
    print(" - Name: %s" % voice.name)
    vlist.append((voice.name,num))
    num += 1

def do_pyttsx3(srt,speed,voice):

    print(srt,speed,voice)

    voice = int(voice)

    generate_audio(path=srt,rate=int(speed),voice_idx=voice)

    return f"output/{vlist[voice][0]}.wav" 


if __name__ == '__main__':
    
    do_pyttsx3("./output/eng.srt",240,3)


