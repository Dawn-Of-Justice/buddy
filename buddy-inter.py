from openai import OpenAI
import pyaudio
import wave
from txt2speech import OpenVoiceProcessor
from speech2txt import STT
from interpreter import interpreter

interpreter.offline = True
interpreter.llm.model = "openai/x"
interpreter.llm.api_key = "fuck_key"
interpreter.llm.api_base = "http://localhost:1234/v1"

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def output_string(segments):
    transcription_segments = []
    for segment in segments:
        transcription_segments.append("%s" % (segment.text))
    return transcription_segments[0] if transcription_segments else ""

class Buddy:
    def __init__(self):
        self.chat_log_filename = "history.txt"
        interpreter.system_message = open_file("prompt.txt")

    def play_audio(self, file_path):
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

    def chatgpt_streamed(self, user_input, conversation_history, bot_name):

        streamed_completion = interpreter.chat(user_input)
        interpreter.messages = conversation_history 

        full_response = ""
        line_buffer = ""

        with open(self.chat_log_filename, "a") as log_file:
            for chunk in streamed_completion:
                delta_content = chunk.choices[0].delta.content

                if delta_content is not None:
                    line_buffer += delta_content

                    if '\n' in line_buffer:
                        lines = line_buffer.split('\n')
                        for line in lines[:-1]:
                            print(line)
                            full_response += line + '\n'
                            log_file.write(f"{bot_name}: {line}\n") 
                        line_buffer = lines[-1]

            if line_buffer:
                print(line_buffer)
                full_response += line_buffer
                log_file.write(f"{bot_name}: {line_buffer}\n")  

        return full_response


if __name__ == "__main__":

    conversation_history = []
    name = 'Buddy'

    whisp = STT()
    while True:
        try:
            voice = whisp.listen()
            segments = whisp.transcribe(voice)
            msg = output_string(segments)
            if msg:
                audio_gen = OpenVoiceProcessor()
                buddy = Buddy()
                print("You:", msg)
                conversation_history.append({"role": "user", "content": msg})
                print(f"{name}:",end='')
                chatbot_response = buddy.chatgpt_streamed(msg, conversation_history, name)
                conversation_history.append({"role": "assistant", "content": chatbot_response})
                
                prompt = chatbot_response
                style = "default"
                reference_speaker = "voice.mp3"
                path = audio_gen.process_audio(prompt, style, reference_speaker)
                buddy.play_audio(path)

                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

        except KeyboardInterrupt:
            break

        except Exception as e:
            print("Error: ", e)


