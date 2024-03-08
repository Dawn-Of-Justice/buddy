from openai import OpenAI
import pyaudio
import wave
from txt2speech import OpenVoiceProcessor
from speech2txt import STT
import json
    
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

chat_log_filename = "history.txt"
allowed_tones = ["friendly", "cheerful", "excited", "sad", "angry", "terrified", "shouting", "whispering", "default"]

def play_audio(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def chatgpt_streamed(user_input, system_message, conversation_history):
    
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input + 'give it in above the JSON format only'}]  
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    full_response = ""

    for chunk in streamed_completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content

    if full_response:
        res_dict = json.loads(full_response)

    return res_dict

def output_string(segments):
    transcription_segments = []
    for segment in segments:
        transcription_segments.append("%s" % (segment.text))
    return transcription_segments[0] if transcription_segments else ""

def writer(response):
    with open(chat_log_filename, "a") as txt_file:
        txt_file.write(response)

def process(array):
    new_list = []
    for each in array:
        parts = each.split(':', 1)
        if len(parts) != 2:
            raise Exception("Bad Format")
        role = "user" if parts[0].strip() == "User" else "assistant"
        content = parts[1].strip()
        new_list.append({"role": role, "content": content})
    return new_list

def valid_LLM_response(msg, system_message, conversation_history):
    while True:
        res = ''
        try:
            res = chatgpt_streamed(msg, system_message, conversation_history)
            return json.dumps(res)
        except:
            continue


if __name__ == "__main__":

    conversation_history = process(read_file(chat_log_filename).split('\n')[-10:])
    system_message = read_file("prompt.txt")
    name = 'Buddy'

    whisp = STT()
    while True:
        try:
            voice = whisp.listen()
            segments = whisp.transcribe(voice)
            msg = output_string(segments)
            if msg:
                audio_gen = OpenVoiceProcessor()
                print("You:", msg)
                conversation_history.append({"role": "user", "content": msg})
                chatbot_response = json.loads(valid_LLM_response(msg, system_message, conversation_history))
                print(f"{name}: {chatbot_response['message']}")
                writer('\nUser : ' + msg + '\n' + name + ':' + json.dumps(chatbot_response))
                conversation_history.append({"role": "assistant", "content": chatbot_response["message"]})
                prompt = chatbot_response["message"]
                if chatbot_response["tone"] not in allowed_tones:
                    chatbot_response["tone"] = "default"
                style = chatbot_response["tone"]
                reference_speaker = "voice.mp3"
                path = audio_gen.process_audio(prompt, style, reference_speaker)
                play_audio(path)

                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

        except KeyboardInterrupt:
            break

        except Exception as e:
            print("Error: ", e)


