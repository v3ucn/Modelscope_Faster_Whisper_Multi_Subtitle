from modelscope.pipelines import pipeline as pipeline_ali
from modelscope.utils.constant import Tasks
from moviepy.editor import VideoFileClip

import os

import ffmpeg

from faster_whisper import WhisperModel
import math

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline

from slicer2 import Slicer

import librosa

import soundfile

from funasr import AutoModel

from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 指定本地目录
local_dir_root = "./models_from_modelscope"

# model_dir_cirm = snapshot_download('damo/speech_frcrn_ans_cirm_16k', cache_dir=local_dir_root)

# model_dir_ins = snapshot_download('damo/nlp_csanmt_translation_en2zh', cache_dir=local_dir_root)


model_dir_cirm = './models_from_modelscope/damo/speech_frcrn_ans_cirm_16k'

model_dir_ins = './models_from_modelscope/damo/nlp_csanmt_translation_en2zh'


device = "cuda" if torch.cuda.is_available() else "cpu"



# 合并字幕
def merge_sub(video_path,srt_path):

    if os.path.exists("./test_srt.mp4"):
        os.remove("./test_srt.mp4")

    ffmpeg.input(video_path).output("./test_srt.mp4", vf="subtitles=" + srt_path).run()

    return "./test_srt.mp4"


def make_tran_ja2zh_neverLife(srt_path):

    model_path = "neverLife/nllb-200-distilled-600M-ja-zh"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="jpn_Jpan", tgt_lang="zho_Hans", from_pt=True)

    # pipe = pipeline(model="larryvrh/mt5-translation-ja_zh")

    with open(srt_path, 'r',encoding="utf-8") as file:
        gweight_data = file.read()

    result = gweight_data.split("\n\n")

    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    for res in result:

        line_srt = res.split("\n")
        
        try:
            # translated_text = pipe(f'<-ja2zh-> {line_srt[2]}')[0]['translation_text']
            # print(translated_text)
            input_ids = tokenizer.encode(line_srt[2], max_length=128, padding=True, return_tensors='pt')
            outputs = model.generate(input_ids, num_beams=4, max_new_tokens=128)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(translated_text)

        except IndexError as e:
            # 处理下标越界异常
            print(f"翻译完毕")
            break
        except Exception as e:
             print(str(e))
             
        
        with open("./two.srt","a",encoding="utf-8")as f:f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    with open("./two.srt","r",encoding="utf-8") as f:
        content = f.read()
    
    return content



def make_tran_ko2zh(srt_path):

    # pipe = pipeline(model="yesj1234/mbart_cycle1_ko-zh",device=device,from_pt=True)

    model_path = "./model_from_hg/ko-zh/"

    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True)

    with open(srt_path, 'r',encoding="utf-8") as file:
        gweight_data = file.read()

    result = gweight_data.split("\n\n")

    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    for res in result:

        line_srt = res.split("\n")
        
        try:

            # translated_text = pipe(f'<-ja2zh-> {line_srt[2]}')[0]['translation_text']
            # print(translated_text)

            input_ids = tokenizer.encode(line_srt[2], max_length=128, padding=True, return_tensors='pt')
            outputs = model.generate(input_ids, num_beams=4, max_new_tokens=128)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(translated_text)

        except IndexError as e:
            # 处理下标越界异常
            print(f"翻译完毕")
            break
        except Exception as e:
             print(str(e))
             
        
        with open("./two.srt","a",encoding="utf-8")as f:f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    with open("./two.srt","r",encoding="utf-8") as f:
        content = f.read()
    
    return content

def make_tran_ja2zh(srt_path):

    # pipe = pipeline(model="larryvrh/mt5-translation-ja_zh",device=device)


    model_path = "./model_from_hg/ja-zh/"

    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True)

    with open(srt_path, 'r',encoding="utf-8") as file:
        gweight_data = file.read()

    result = gweight_data.split("\n\n")

    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    for res in result:

        line_srt = res.split("\n")
        
        try:

            # translated_text = pipe(f'<-ja2zh-> {line_srt[2]}')[0]['translation_text']
            # print(translated_text)

            input_ids = tokenizer.encode(f'<-ja2zh-> {line_srt[2]}', max_length=128, padding=True, return_tensors='pt')
            outputs = model.generate(input_ids, num_beams=4, max_new_tokens=128)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(translated_text)



        except IndexError as e:
            # 处理下标越界异常
            print(f"翻译完毕")
            break
        except Exception as e:
             print(str(e))
             
        
        with open("./two.srt","a",encoding="utf-8")as f:f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    with open("./two.srt","r",encoding="utf-8") as f:
        content = f.read()
    
    return content


def make_tran_zh2en(srt_path):

    model_path = "./model_from_hg/zh-en/" 

    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True)

    with open(srt_path, 'r',encoding="utf-8") as file:
        gweight_data = file.read()

    result = gweight_data.split("\n\n")

    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    for res in result:

        line_srt = res.split("\n")
        try:

            tokenized_text = tokenizer.prepare_seq2seq_batch([line_srt[2]], return_tensors='pt')
            translation = model.generate(**tokenized_text)
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
            translated_text = translated_text.replace("<pad>","").replace("</s>","").strip()
            print(translated_text)

        except IndexError as e:
            # 处理下标越界异常
            print(f"翻译完毕")
            break
        except Exception as e:
             print(str(e))
             
        
        with open("./two.srt","a",encoding="utf-8")as f:f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    with open("./two.srt","r",encoding="utf-8") as f:
        content = f.read()
    
    return content


# 翻译字幕 英译中
def make_tran(srt_path):


    model_path = "./model_from_hg/en-zh/"

    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True)

    with open(srt_path, 'r',encoding="utf-8") as file:
        gweight_data = file.read()

    result = gweight_data.split("\n\n")

    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    for res in result:

        line_srt = res.split("\n")
        try:

            tokenized_text = tokenizer.prepare_seq2seq_batch([line_srt[2]], return_tensors='pt')
            translation = model.generate(**tokenized_text)
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
            translated_text = translated_text.replace("<pad>","").replace("</s>","").strip()
            print(translated_text)

        except IndexError as e:
            # 处理下标越界异常
            print(f"翻译完毕")
            break
        except Exception as e:
             print(str(e))
             
        
        with open("./two.srt","a",encoding="utf-8")as f:f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    with open("./two.srt","r",encoding="utf-8") as f:
        content = f.read()

    return content

# # 翻译字幕
# def make_tran_ali():

#     pipeline_ins = pipeline(task=Tasks.translation, model=model_dir_ins)

#     with open("./video.srt", 'r',encoding="utf-8") as file:
#         gweight_data = file.read()

#     result = gweight_data.split("\n\n")

#     if os.path.exists("./two.srt"):
#         os.remove("./two.srt")

#     for res in result:

#         line_srt = res.split("\n")
#         try:
#             outputs = pipeline_ins(input=line_srt[2])
#             print(outputs['translation'])
#         except IndexError as e:
#             # 处理下标越界异常
#             print(f"翻译完毕")
#             break
#         except Exception as e:
#              print(str(e))
             
        
#         with open("./two.srt","a",encoding="utf-8")as f:f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{outputs['translation']}\n\n")

#     return "翻译完毕"



def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = math.floor((seconds % 1) * 1000)
    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
    return output


emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "",
	"<|SAD|>": "",
	"<|ANGRY|>": "",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "",
	"<|Speech|>": "",
	"<|Applause|>": "",
	"<|Laughter|>": "",
	"<|FEARFUL|>": "",
	"<|DISGUSTED|>": "",
	"<|SURPRISED|>": "",
	"<|Cry|>": "",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "",
	"<|Breath|>": "",
	"<|Cough|>": "",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
            "ko": "KO|",
            "yue": "YUE|",
        }

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def ms_to_srt_time(ms):
    N = int(ms)
    hours, remainder = divmod(N, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    timesrt = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    # print(timesrt)
    return timesrt

def time_to_srt(time_in_seconds):
    """
    将秒数转换为 SRT 时间戳格式。

    Args:
        time_in_seconds: 秒数。

    Returns:
        一个 SRT 时间戳字符串。
    """
    milliseconds = int(time_in_seconds * 1000)
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds = (milliseconds % 60000) // 1000
    milliseconds = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
# 制作字幕文件 阿里
def make_srt_sv(file_path):


    model_dir = "iic/SenseVoiceSmall"
    input_file = (file_path)

    model = AutoModel(model=model_dir,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    trust_remote_code=True, device="cuda:0")

    res = model.generate(
        input=input_file,
        cache={},
        language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        batch_size_s=0, 
    )

    print(res)
    text = res[0]["text"]
    # text = format_str_v3(text)
    text = rich_transcription_postprocess(text)

    print(text)

    return text


    # for filename in os.listdir("./wavs"):
    #     if filename.endswith(".wav"):
    #         filepath = os.path.join("./wavs/", filename)
    #         try:
    #             if os.path.isfile(filepath):
    #                 os.remove(filepath)
    #                 print(f"已删除文件: {filepath}")
    #         except Exception as e:
    #             print(f"删除文件时出错: {filepath} - {e}")

    # # 第一步，先切片

    # audio, sr = librosa.load(file_path, sr=None, mono=False)

    # # 创建Slicer对象
    # slicer = Slicer(
    #     sr=sr,
    #     threshold=-40,
    #     min_length=1500,
    #     min_interval=300,
    #     hop_size=1,
    #     max_sil_kept=150000
    # )

    # # 切割音频
    # chunks = slicer.slice(audio)
    # for i, chunk in enumerate(chunks):
    #     if len(chunk.shape) > 1:
    #         chunk = chunk.T  # Swap axes if the audio is stereo.
    #     soundfile.write(f'./wavs/chunk_{i}.wav', chunk, sr)


    # srtlines = []
    # audio_samples = 0
    # audio_opt = []
    # for filename in os.listdir("./wavs"):
    #     if filename.endswith(".wav"):
    #         filepath = os.path.join("./wavs/", filename)
    #         print(filepath)

    #         model_dir = "iic/SenseVoiceSmall"
    #         input_file = (filepath)

    #         model = AutoModel(model=model_dir,
    #                         vad_model="fsmn-vad",
    #                         vad_kwargs={"max_single_segment_time": 30000},
    #                         trust_remote_code=True, device="cuda:0")

    #         res = model.generate(
    #             input=input_file,
    #             cache={},
    #             language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
    #             use_itn=False,
    #             batch_size_s=0, 
    #         )

    #         # print(res)
    #         text = res[0]["text"]
    #         # text = format_str_v3(text)
    #         text = rich_transcription_postprocess(text)

    #         print(text)

    #         audio, sampling_rate = soundfile.read(filepath)

    #         audio_opt.append(audio)

    #         srtline_begin=ms_to_srt_time(audio_samples*1000.0 / sampling_rate)
    #         audio_samples += audio.size
    #         srtline_end=ms_to_srt_time(audio_samples*1000.0 / sampling_rate)

    #         srtlines.append(f"{len(audio_opt)}\n")
    #         srtlines.append(srtline_begin+' --> '+srtline_end+"\n")

    #         srtlines.append(text+"\n\n")

            # exit(-1)

    with open('./video.srt', 'w', encoding='utf-8') as f:
        f.writelines(srtlines)

    with open("./video.srt","r",encoding="utf-8") as f:
        content = f.read()
        
    

    return content
# 制作字幕文件
def make_srt(file_path,model_name="small"):

    
    # if device == "cuda":
    #     model = WhisperModel(model_name, device="cuda", compute_type="float16",download_root="./model_from_whisper",local_files_only=False)
    # else:
    #     model = WhisperModel(model_name, device="cpu", compute_type="int8",download_root="./model_from_whisper",local_files_only=False)
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        
    if device == "cuda":
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16",download_root="./model_from_whisper",local_files_only=False)
        except Exception as e:
            model = WhisperModel(model_name, device="cuda", compute_type="int8_float16",download_root="./model_from_whisper",local_files_only=False)
    else:
        model = WhisperModel(model_name, device="cpu", compute_type="int8",download_root="./model_from_whisper",local_files_only=False)

    segments, info = model.transcribe(file_path, beam_size=5,vad_filter=True,vad_parameters=dict(min_silence_duration_ms=500))

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    count = 0
    with open('./video.srt', 'w',encoding="utf-8") as f:  # Open file for writing
        for segment in segments:
            count +=1
            duration = f"{convert_seconds_to_hms(segment.start)} --> {convert_seconds_to_hms(segment.end)}\n"
            text = f"{segment.text.lstrip()}\n\n"
            
            f.write(f"{count}\n{duration}{text}")  # Write formatted string to the file
            print(f"{duration}{text}",end='')

    with open("./video.srt","r",encoding="utf-8") as f:
        content = f.read()

    return content



# 提取人声
def movie2audio(video_path):

    # 读取视频文件
    video = VideoFileClip(video_path)

    # 提取视频文件中的声音
    audio = video.audio

    # 将声音保存为WAV格式
    audio.write_audiofile("./audio.wav")

    ans = pipeline_ali(
        Tasks.acoustic_noise_suppression,
        model=model_dir_cirm)
    
    ans('./audio.wav',output_path='./output.wav')

    return "./output.wav"










