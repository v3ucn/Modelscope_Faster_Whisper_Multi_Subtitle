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


initial_md = """

项目地址:https://github.com/v3ucn/Modelscope_Faster_Whisper_Multi_Subtitle

作者：刘悦的技术博客  https://space.bilibili.com/3031494

"""

def do_pyttsx3(srt,speed,voice):

    voice = int(voice)

    generate_audio(path=srt,rate=int(speed),voice_idx=voice)

    return f"output/{vlist[voice][0]}.wav" 

def do_speech(video):

    res = movie2audio(video)

    return res


def do_trans_video(model_type,video_path):

    srt_text = make_srt(video_path,model_type)

    return srt_text

def do_trans_video_sv(video_path):

    srt_text = make_srt_sv(video_path)

    return srt_text

def do_trans_audio(model_type):

    srt_text = make_srt(f'{ROOT_DIR}/audio.wav',model_type)

    return srt_text

def do_trans_en2zh(srt_path):

    return make_tran(srt_path)


def do_trans_en2zh_deep(srt_path):

    return make_tran_deep(srt_path,"EN","ZH")

def do_trans_zh2en_deep(srt_path):

    return make_tran_deep(srt_path,"ZH","EN")

def do_trans_zh2ja_deep(srt_path):

    return make_tran_deep(srt_path,"ZH","JA")

def do_trans_zh2ko_deep(srt_path):

    return make_tran_deep(srt_path,"ZH","KO")

def do_trans_ja2zh_deep(srt_path):

    return make_tran_deep(srt_path,"JA","ZH")

def do_trans_ko2zh_deep(srt_path):

    return make_tran_deep(srt_path,"KO","ZH")




def do_trans_en2zh_qwen2(model_path_qwen2,srt_path):

    return make_tran_qwen2(model_path_qwen2,srt_path,"zh")

def do_trans_zh2en_qwen2(model_path_qwen2,srt_path):

    return make_tran_qwen2(model_path_qwen2,srt_path,"en")

def do_trans_ja2zh_qwen2(model_path_qwen2,srt_path):

    return make_tran_qwen2(model_path_qwen2,srt_path,"zh")

def do_trans_ko2zh_qwen2(model_path_qwen2,srt_path):

    return make_tran_qwen2(model_path_qwen2,srt_path,"zh")

def do_trans_zh2en(srt_path):

    return make_tran_zh2en(srt_path)

def do_trans_ja2zh(srt_path):

    return make_tran_ja2zh(srt_path)

def do_trans_ko2zh(srt_path):

    return make_tran_ko2zh(srt_path)

def do_srt_sin(video_path):

    return merge_sub(video_path,f"{ROOT_DIR}/output/video.srt")

def do_srt_two(video_path):

    return merge_sub(video_path,f"{ROOT_DIR}/output/two.srt")


def do_srt_two_single(video_path):

    return merge_sub(video_path,f"{ROOT_DIR}/output/two_single.srt")


def save_srt(text):

    with open(rf'{ROOT_DIR}/output/video.srt','w',encoding='utf-8') as f:
        f.write(text + "\n")

    gr.Info('字幕文件修改成功,字幕保存在output目录')


def save_two(text,text_2):

    with open(rf'{ROOT_DIR}/output/two.srt','w',encoding='utf-8') as f:
        f.write(text + "\n")

    with open(rf'{ROOT_DIR}/output/two_single.srt','w',encoding='utf-8') as f:
        f.write(text_2 + "\n")

    gr.Info('字幕文件修改成功,字幕保存在output目录')
    
    



with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("视频处理(Video)"):
        with gr.Row():

            ori_video = gr.Video(label="请上传视频(Upload Video)")
        
            speech_button = gr.Button("提取人声(如果视频没有背景音也可以不做)Extract human voice (you don't have to do it if the video has no background sound)")

            speech_audio = gr.Audio(label="提取的人声(Extract voice)")

    
    speech_button.click(do_speech,inputs=[ori_video],outputs=[speech_audio])
    
    with gr.Accordion("转写字幕"):

        with gr.Row():
            with gr.Column():
                
                # model_type = gr.Dropdown(choices=["small","medium","large-v3","large-v2"], value="small", label="选择faster_Whisper模型/Select faster_Whisper model",interactive=True)

                model_type = gr.Textbox(label="填写faster_Whisper模型/Fill in the faster_Whisper model,也可以填写small,medium,large,large-v2,large-v3,faster-whisper-large-v3-turbo-ct2,模型越大，速度越慢，但字幕的准确度越高，酌情填写，用文本框是因为你可以填写其他huggingface上的开源模型地址",value="faster-whisper-large-v3-turbo-ct2")

        # with gr.Row():
        #     with gr.Column():
                
        #         language = gr.Dropdown(["ja", "en", "zh","ko","yue"], value="zh", label="选择转写的语言",interactive=True)


        with gr.Row():
            
            transcribe_button_whisper = gr.Button("Whisper视频直接转写字幕(Video direct rewriting subtitles)")

            transcribe_button_audio = gr.Button("Whisper提取人声转写字幕(Extract voice transliteration subtitles)")


            # transcribe_button_video_sv = gr.Button("阿里SenseVoice视频直接转写字幕")

            result1 = gr.Textbox(label="字幕結果(会在项目目录生成video.srt/video.srt is generated in the current directory)",value=" ",interactive=True)

            transcribe_button_audio_save = gr.Button("保存字幕修改结果")

        transcribe_button_whisper.click(do_trans_video,inputs=[model_type,ori_video],outputs=[result1])

        transcribe_button_audio_save.click(save_srt,inputs=[result1],outputs=[])

        # transcribe_button_video_sv.click(do_trans_video_sv,inputs=[ori_video],outputs=[result1])

        transcribe_button_audio.click(do_trans_audio,inputs=[model_type],outputs=[result1])


    # with gr.Accordion("HuggingFace大模型字幕翻译"):
    #     with gr.Row():


    #         srt_path = gr.Textbox(label="原始字幕地址，默认为项目目录中的video.srt,也可以输入其他路径",value="./video.srt")

    #         trans_button_en2zh = gr.Button("翻译英语字幕为中文/Translate English subtitles into Chinese")

    #         trans_button_zh2en = gr.Button("翻译中文字幕为英文/Translate Chinese subtitles into English")

    #         trans_button_ja2zh = gr.Button("翻译日文字幕为中文/Translate Japanese subtitles into Chinese")

    #         trans_button_ko2zh = gr.Button("翻译韩文字幕为中文/Translate Korea subtitles into Chinese")

    #         result2 = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)")

    #     trans_button_en2zh.click(do_trans_en2zh,[srt_path],outputs=[result2])

    #     trans_button_zh2en.click(do_trans_zh2en,[srt_path],outputs=[result2])

    #     trans_button_ja2zh.click(do_trans_ja2zh,[srt_path],outputs=[result2])

    #     trans_button_ko2zh.click(do_trans_ko2zh,[srt_path],outputs=[result2])

    with gr.Accordion("Qwen2大模型字幕翻译"):
        with gr.Row():


            srt_path_qwen2 = gr.Textbox(label="原始字幕地址，默认为项目目录中的output/video.srt,也可以输入其他路径",value=f"{ROOT_DIR}/output/video.srt")

            model_path_qwen2 = gr.Textbox(label="ollama中模型名称",value="qwen2:7b")

            trans_button_en2zh_qwen2 = gr.Button("翻译英语字幕为中文/Translate English subtitles into Chinese")

            trans_button_zh2en_qwen2 = gr.Button("翻译中文字幕为英文/Translate Chinese subtitles into English")

            trans_button_ja2zh_qwen2 = gr.Button("翻译日文字幕为中文/Translate Japanese subtitles into Chinese")

            trans_button_ko2zh_qwen2 = gr.Button("翻译韩文字幕为中文/Translate Korea subtitles into Chinese")

        with gr.Row():

            result2 = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)",value=" ",interactive=True)

            result3 = gr.Textbox(label="翻译结果(会在项目目录生成two_single.srt)",value=" ",interactive=True)

            trans_button_ko2zh_qwen2_save = gr.Button("保存修改结果")

        trans_button_en2zh_qwen2.click(do_trans_en2zh_qwen2,[model_path_qwen2,srt_path_qwen2],outputs=[result2,result3])

        trans_button_zh2en_qwen2.click(do_trans_zh2en_qwen2,[model_path_qwen2,srt_path_qwen2],outputs=[result2,result3])

        trans_button_ja2zh_qwen2.click(do_trans_ja2zh_qwen2,[model_path_qwen2,srt_path_qwen2],outputs=[result2,result3])

        trans_button_ko2zh_qwen2.click(do_trans_ko2zh_qwen2,[model_path_qwen2,srt_path_qwen2],outputs=[result2,result3])

        trans_button_ko2zh_qwen2_save.click(save_two,[result2,result3],outputs=[])


    with gr.Accordion("Deepl字幕翻译"):
        with gr.Row():


            srt_path_deep = gr.Textbox(label="原始字幕地址，默认为项目目录中的output/video.srt,也可以输入其他路径",value=f"{ROOT_DIR}/output/video.srt")

            trans_button_en2zh_deep = gr.Button("翻译英语字幕为中文/Translate English subtitles into Chinese")

            trans_button_zh2en_deep = gr.Button("翻译中文字幕为英文/Translate Chinese subtitles into English")

            trans_button_zh2ja_deep = gr.Button("翻译中文字幕为日文/Translate Chinese subtitles into Japanese")

            trans_button_zh2ko_deep = gr.Button("翻译中文字幕为韩文/Translate Chinese subtitles into Korea")

            trans_button_ja2zh_deep = gr.Button("翻译日文字幕为中文/Translate Japanese subtitles into Chinese")

            trans_button_ko2zh_deep = gr.Button("翻译韩文字幕为中文/Translate Korea subtitles into Chinese")

        with gr.Row():

            result2_deep = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)",value=" ",interactive=True)

            result3_deep = gr.Textbox(label="翻译结果(会在项目目录生成two_single.srt)",value=" ",interactive=True)

            trans_button_ko2zh_deep_save = gr.Button("保存修改结果")

        

        trans_button_ko2zh_deep_save.click(save_two,[result2_deep,result3_deep],outputs=[])


    with gr.Accordion("字幕配音(pyttsx3)"):
        with gr.Row():

            srt_path_pyttsx3 = gr.Textbox(label="字幕地址,也可以输入其他路径",value=f"{ROOT_DIR}/output/two_single.srt",interactive=True)

            speed_pyttsx3 = gr.Textbox(label="配音语速(很重要,否则会引起时间轴错乱的问题)",value="240")

            voice_pyttsx3 = gr.Dropdown(choices=vlist,value=3,label="配音的音色选择",interactive=True)

            button_pyttsx3 = gr.Button("生成配音")

            pyttsx3_audio = gr.Audio(label="配音的结果")


    trans_button_en2zh_deep.click(do_trans_en2zh_deep,[srt_path_deep],outputs=[result2_deep,result3_deep,srt_path_pyttsx3])

    trans_button_zh2ja_deep.click(do_trans_zh2ja_deep,[srt_path_deep],outputs=[result2_deep,result3_deep,srt_path_pyttsx3])

    trans_button_zh2en_deep.click(do_trans_zh2en_deep,[srt_path_deep],outputs=[result2_deep,result3_deep,srt_path_pyttsx3])

    trans_button_zh2ko_deep.click(do_trans_zh2ko_deep,[srt_path_deep],outputs=[result2_deep,result3_deep,srt_path_pyttsx3])

    trans_button_ja2zh_deep.click(do_trans_ja2zh_deep,[srt_path_deep],outputs=[result2_deep,result3_deep,srt_path_pyttsx3])

    trans_button_ko2zh_deep.click(do_trans_ko2zh_deep,[srt_path_deep],outputs=[result2_deep,result3_deep,srt_path_pyttsx3])


    button_pyttsx3.click(do_pyttsx3,inputs=[srt_path_pyttsx3,speed_pyttsx3,voice_pyttsx3],outputs=[pyttsx3_audio])

            

    with gr.Accordion("字幕合并"):
        with gr.Row():


            srt_button_sin = gr.Button("将单语字幕合并到视频/Merge monolingual subtitles into video")

            srt_button_two = gr.Button("将双语字幕合并到视频/Merge bilingual subtitles into video")

            srt_button_two_single = gr.Button("将翻译的单语字幕合并到视频")

            result3 = gr.Video(label="带字幕视频")

    srt_button_sin.click(do_srt_sin,inputs=[ori_video],outputs=[result3])
    srt_button_two.click(do_srt_two,inputs=[ori_video],outputs=[result3])
    srt_button_two.click(do_srt_two_single,inputs=[ori_video],outputs=[result3])


    

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

app.queue()
app.launch(inbrowser=True, server_name=args.server_name)
