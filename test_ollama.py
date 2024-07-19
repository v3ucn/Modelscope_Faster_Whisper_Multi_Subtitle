import ollama


response = ollama.chat(model='qwen2:7b',messages=[
{
'role':'user',
'content':'"you fucked up , bitch" 翻译为中文，只给我文本的翻译，别添加其他的内容，因为我要做字幕，谢谢'
}])
print(response['message']['content'])