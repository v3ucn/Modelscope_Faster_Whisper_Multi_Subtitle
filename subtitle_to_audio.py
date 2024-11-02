import os
import tempfile
import argparse
from pysubparser import parser
from pydub import AudioSegment
import pyttsx3

def time_to_ms(time):
  return ((time.hour * 60 + time.minute) * 60 + time.second) * 1000 + time.microsecond / 1000

def generate_audio(path, rate=200, voice_idx=0):  
  print("Generating audio file for {} with {}".format(path, "pyttsx3"))      

  subtitles = parser.parse(path)

  tts_engine = pyttsx3.init()
  tts_engine.setProperty('rate', rate)
  tts_engine.setProperty('voice', tts_engine.getProperty('voices')[voice_idx].id)

  audio_sum = AudioSegment.empty()   
  
  with tempfile.TemporaryDirectory() as tmpdirname:          
    print('created temporary directory', tmpdirname)            

    temp_file_path = os.path.join(tmpdirname, "temp.wav")
    prev_subtitle = None
    prev_audio_duration_ms = 0
    for subtitle in subtitles:   
      tts_engine.save_to_file(subtitle.text, temp_file_path)
      tts_engine.runAndWait()

      audio_segment = AudioSegment.from_wav(temp_file_path)         

      print(subtitle.start, subtitle.text)
      
      if prev_subtitle is None:
        silence_duration_ms = time_to_ms(subtitle.start)
      else:
        silence_duration_ms = time_to_ms(subtitle.start) - time_to_ms(prev_subtitle.start) - prev_audio_duration_ms

      audio_sum = audio_sum + AudioSegment.silent(duration=silence_duration_ms) + audio_segment                   
      
      prev_subtitle = subtitle
      prev_audio_duration_ms = len(audio_segment)

    with open('output/pyttsx3.wav', 'wb') as out_f:
      audio_sum.export(out_f, format='wav')      

if __name__ == "__main__":      
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("-p", "--path", help="subtitle file path",default="two_single.srt")
  arg_parser.add_argument("-r", "--rate", help="speech rate(words per minute)", type=int, default=240)
  arg_parser.add_argument("-v", "--voice-idx", help="voice selection", type=int, default=1, choices=[0, 1])
  
  args = arg_parser.parse_args()
  
  generate_audio(path=args.path, rate=args.rate, voice_idx=args.voice_idx)    

