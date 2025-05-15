from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import os

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

snac_device = os.environ.get("SNAC_DEVICE", "xla" if "xla" in torch.backends.__dict__ else ("cuda" if torch.cuda.is_available() else "cpu"))
model = model.to(snac_device)

def convert_to_audio(multiframe, count):
    if len(multiframe) < 7:
        return

    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    for j in range(num_frames):
        i = 7 * j
        codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])
        codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1], frame[i+4]], device=snac_device, dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2], frame[i+3], frame[i+5], frame[i+6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    if any(torch.any(c < 0) or torch.any(c > 4096) for c in codes):
        return

    with torch.inference_mode():
        audio_hat = model.decode(codes)

    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()

def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")

    if last_token_start == -1:
        print("No token found in the string")
        return None

    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None

async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen):
    audio_queue = queue.Queue()

    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()