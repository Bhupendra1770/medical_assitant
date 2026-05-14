import asyncio
import edge_tts
import os
import uuid

async def generate_tts_webm(
    text: str,
    voice: str = "en-IN-NeerjaNeural",
    rate: str = "+10%",
) -> str:
    output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tts_outputs")
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    file_id = str(uuid.uuid4())
    mp3_path = os.path.join(output_folder, f"{file_id}.mp3")
    webm_path = os.path.join(output_folder, f"{file_id}.webm")

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(mp3_path)

    ret = os.system(f'ffmpeg -y -i "{mp3_path}" "{webm_path}" 2>/dev/null')
    if ret != 0:
        return mp3_path

    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    return webm_path
