import argparse
import os

import gradio as gr

from utils import movie2audio,make_srt,make_tran,merge_sub,make_tran_zh2en,make_tran_ja2zh




initial_md = """


作者：刘悦的技术博客  https://space.bilibili.com/3031494

"""


def do_speech(video):

    res = movie2audio(video)

    return res


def do_trans_video(model_type,video_path):

    srt_text = make_srt(video_path,model_type)

    return srt_text

def do_trans_audio(model_type):

    srt_text = make_srt('./audio.wav',model_type)

    return srt_text

def do_trans_en2zh():

    return make_tran()

def do_trans_zh2en():

    return make_tran_zh2en()

def do_trans_ja2zh():

    return make_tran_ja2zh()

def do_srt_sin(video_path):

    return merge_sub(video_path,"./video.srt")

def do_srt_two(video_path):

    return merge_sub(video_path,"./two.srt")



with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("视频(Video)"):
        with gr.Row():

            ori_video = gr.Video(label="请上传视频(Upload Video)", type="filepath").style(width=512)
        
            speech_button = gr.Button("提取人声(如果视频没有背景音也可以不做)Extract human voice (you don't have to do it if the video has no background sound)")

            speech_audio = gr.Audio(label="提取的人声(Extract voice)")

    
    speech_button.click(do_speech,inputs=[ori_video],outputs=[speech_audio])
    

    with gr.Row():
        with gr.Column():
            

            # model_type = gr.Dropdown(choices=["small","medium","large-v3","large-v2"], value="small", label="选择faster_Whisper模型/Select faster_Whisper model",interactive=True)

            model_type = gr.Textbox(label="填写faster_Whisper模型/Fill in the faster_Whisper model,也可以填写small,medium,large,large-v2,large-v3,模型越大，速度越慢，但字幕的准确度越高，酌情填写，用文本框是因为你可以填写其他huggingface上的开源模型地址",value="medium")

            
        transcribe_button_whisper = gr.Button("视频直接转写字幕(Video direct rewriting subtitles)")

        transcribe_button_audio = gr.Button("提取人声转写字幕(Extract voice transliteration subtitles)")

        result1 = gr.Textbox(label="字幕結果(会在项目目录生成video.srt/video.srt is generated in the current directory)")

    transcribe_button_whisper.click(do_trans_video,inputs=[model_type,ori_video],outputs=[result1])

    transcribe_button_audio.click(do_trans_audio,inputs=[model_type],outputs=[result1])


    with gr.Row():


        trans_button_en2zh = gr.Button("翻译英语字幕为中文/Translate English subtitles into Chinese")

        trans_button_zh2en = gr.Button("翻译中文字幕为英文/Translate Chinese subtitles into English")

        trans_button_ja2zh = gr.Button("翻译日文字幕为中文/Translate Japanese subtitles into Chinese")

        result2 = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)")

    trans_button_en2zh.click(do_trans_en2zh,outputs=[result2])

    trans_button_zh2en.click(do_trans_zh2en,outputs=[result2])

    trans_button_ja2zh.click(do_trans_ja2zh,outputs=[result2])

    with gr.Row():


        srt_button_sin = gr.Button("将单语字幕合并到视频/Merge monolingual subtitles into video")

        srt_button_two = gr.Button("将双语字幕合并到视频/Merge bilingual subtitles into video")

        result3 = gr.Video(label="带字幕视频", type="filepath").style(width=512)

    srt_button_sin.click(do_srt_sin,inputs=[ori_video],outputs=[result3])
    srt_button_two.click(do_srt_two,inputs=[ori_video],outputs=[result3])


    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-name",
    type=str,
    default=None,
    help="Server name for Gradio app",
)
parser.add_argument(
    "--no-autolaunch",
    action="store_true",
    default=False,
    help="Do not launch app automatically",
)
args = parser.parse_args()

app.launch(inbrowser=not args.no_autolaunch, server_name=args.server_name)
