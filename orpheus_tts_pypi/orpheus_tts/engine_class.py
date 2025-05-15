import asyncio
import torch
import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import threading
import queue
from .decoder import tokens_decoder_sync

class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16, tokenizer='canopylabs/orpheus-3b-0.1-pretrained', **engine_kwargs):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path):
        try:
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return AutoTokenizer.from_pretrained("gpt2")

    def _map_model_params(self, model_name):
        model_map = {
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if model_name in unsupported_models:
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported.")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name

    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            **self.engine_kwargs
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    def validate_voice(self, voice):
        if voice and voice not in self.available_voices:
            raise ValueError(f"Voice {voice} is not available for model {self.model_name}")

    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>" if voice else f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            prompt_tokens = self.tokenizer(f"{voice}: {prompt}" if voice else prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            return self.tokenizer.decode(all_input_ids[0])

    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        prompt_string = self._format_prompt(prompt, voice)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))