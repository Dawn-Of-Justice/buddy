from openai import OpenAI
import pyaudio
import wave
from txt2speech import OpenVoiceProcessor
from speech2txt import STT
    
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

chat_log_filename = "history.txt"

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

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def chatgpt_streamed(user_input, system_message, conversation_history):
    
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True
    )

    full_response = ""

    for chunk in streamed_completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content

    print(full_response)

    return full_response

def output_string(segments):
    transcription_segments = []
    for segment in segments:
        transcription_segments.append("%s" % (segment.text))
    return transcription_segments[0] if transcription_segments else ""


if __name__ == "__main__":

    conversation_history = []
    system_message = open_file("prompt.txt")
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
                print(f"{name}:",end='')
                chatbot_response = chatgpt_streamed(msg, system_message, conversation_history, name)
                conversation_history.append({"role": "assistant", "content": chatbot_response})
                
                prompt = chatbot_response
                style = "default"
                reference_speaker = "voice.mp3"
                path = audio_gen.process_audio(prompt, style, reference_speaker)
                play_audio(path)

                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

        except KeyboardInterrupt:
            break

        except Exception as e:
            print("Error: ", e)


