import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import warnings
warnings.filterwarnings("ignore")

reference_speaker = 'openvoice/voice.mp3'

class OpenVoiceProcessor:
    
    def __init__(self):
        self.ckpt_base = 'openvoice/checkpoints/base_speakers/EN'
        self.ckpt_converter = 'openvoice/checkpoints/converter'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = 'openvoice/outputs'
        self.base_speaker_tts = BaseSpeakerTTS(f'{self.ckpt_base}/config.json', device=self.device)
        self.base_speaker_tts.load_ckpt(f'{self.ckpt_base}/checkpoint.pth')
        self.tone_color_converter = ToneColorConverter(f'{self.ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{self.ckpt_converter}/checkpoint.pth')
        os.makedirs(self.output_dir, exist_ok=True)
        self.source_se = torch.load(f'{self.ckpt_base}/en_default_se.pth').to(device=self.device)

    def process_audio(self,prompt, style ,reference_speaker):
        target_se, audio_name = se_extractor.get_se(reference_speaker, self.tone_color_converter, target_dir='processed', vad=True)
        save_path = f'{self.output_dir}/output_en_default.wav'
        src_path = f'{self.output_dir}/tmp.wav'

        self.base_speaker_tts.tts(prompt, src_path, speaker=style, language='English', speed=1.0)

        encode_message = "@MyShell"
        self.tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=self.source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        
        return src_path
    
if __name__ == "__main__":

    style = 'default'
    prompt = "Hello"
    open_voice_processor = OpenVoiceProcessor()
    open_voice_processor.process_audio(prompt, style, reference_speaker)