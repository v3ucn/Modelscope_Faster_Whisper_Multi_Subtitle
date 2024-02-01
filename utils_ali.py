from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from moviepy.editor import VideoFileClip

model_dir_cirm = './models_from_modelscope/damo/speech_frcrn_ans_cirm_16k'


# 提取人声
def movie2audio(video_path):

    # 读取视频文件
    video = VideoFileClip(video_path)

    # 提取视频文件中的声音
    audio = video.audio

    # 将声音保存为WAV格式
    audio.write_audiofile("./audio.wav")

    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_dir_cirm)
    
    ans('./audio.wav',output_path='./output.wav')

    return "./output.wav"