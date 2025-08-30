#!/usr/bin/env python3
"""
Concurrent batched inference test.

This script starts a small FastAPI server that batches incoming text generation
requests (within a short time window) and performs a single batched `.generate()`
call on the model. A client section then sends N concurrent requests to that
server and records latency/throughput metrics.

Run: python concurrent_batched_server_test.py
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import time
from typing import List

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class QueryRequest(BaseModel):
    id: int
    text: str
    max_new_tokens: int = 128


def start_server_process(model_dir: str, adapter_dir: str, host: str, port: int, batch_wait_ms: int, max_batch: int):
    """Spawn a process that runs the FastAPI server."""

    def _run():
        app = FastAPI()

        request_queue: asyncio.Queue = asyncio.Queue()

        @app.on_event("startup")
        async def startup():
            print('Server startup: loading tokenizer and model...')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            app.state.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            if app.state.tokenizer.pad_token_id is None:
                app.state.tokenizer.pad_token = app.state.tokenizer.eos_token

            try:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
                base = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb, device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True)
                print('Server: loaded base model with 4-bit quantization')
            except Exception as e:
                print('Server: quantized load failed, loading fp16:', e)
                base = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)

            app.state.model = PeftModel.from_pretrained(base, adapter_dir)
            app.state.model.eval()
            print('Server: model + adapter loaded')

            # container for batch metrics collected during runtime
            app.state.batch_metrics = []
            app.state.batch_counter = 0

            app.state.batch_task = asyncio.create_task(batch_worker(app, request_queue, batch_wait_ms / 1000.0, max_batch))

        @app.on_event("shutdown")
        async def shutdown():
            try:
                app.state.batch_task.cancel()
            except Exception:
                pass

        @app.get('/health')
        async def health():
            return {'status': 'ok'}

        @app.post('/generate')
        async def generate(req: QueryRequest):
            fut = asyncio.get_event_loop().create_future()
            await request_queue.put((req.id, req.text, req.max_new_tokens, fut))
            try:
                result = await asyncio.wait_for(fut, timeout=300)
                return result
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail='Generation timed out')

        uvicorn.run(app, host=host, port=port, log_level='info')

    p = mp.Process(target=_run, daemon=True)
    p.start()
    return p


async def batch_worker(app: FastAPI, queue: asyncio.Queue, wait_s: float, max_batch: int):
    tokenizer = app.state.tokenizer
    model = app.state.model
    device = next(model.parameters()).device

    while True:
        item = await queue.get()
        batch = [item]
        t0 = time.time()

        while len(batch) < max_batch:
            elapsed = time.time() - t0
            remaining = max(0, wait_s - elapsed)
            try:
                more = await asyncio.wait_for(queue.get(), timeout=remaining)
                batch.append(more)
            except asyncio.TimeoutError:
                break

        ids, texts, max_toks, futures = zip(*[(it[0], it[1], it[2], it[3]) for it in batch])

        # tokenize and move to device
        inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        input_lengths = attention_mask.sum(dim=1).cpu().tolist()

        # record GPU memory before generation
        gpu_mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        batch_start = time.time()
        # assign a batch id for correlation
        batch_id = app.state.batch_counter
        app.state.batch_counter += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            generated = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max(max_toks), do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_end = time.time()
        batch_time = batch_end - batch_start
        gpu_mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        batch_size = len(batch)

        # persist batch-level metrics in app state for post-run analysis
        try:
            app.state.batch_metrics.append({
                'batch_id': batch_id,
                'timestamp': batch_start,
                'batch_time': batch_time,
                'batch_size': batch_size,
                'gpu_mem_before': gpu_mem_before,
                'gpu_mem_after': gpu_mem_after,
            })
        except Exception:
            pass

        for i, fut in enumerate(futures):
            input_len = input_lengths[i]
            out = generated[i]
            new_tokens = out[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            fut.set_result({
                'id': ids[i],
                'text': text,
                'input_tokens': input_len,
                'generated_tokens': len(new_tokens),
                'batch_time': batch_time,
                'batch_size': batch_size,
                'gpu_mem_before': gpu_mem_before,
                'gpu_mem_after': gpu_mem_after,
                'batch_id': batch_id,
            })


async def run_client(host: str, port: int, queries: List[str]):
    url = f'http://{host}:{port}/generate'
    async with aiohttp.ClientSession() as sess:
        tasks = []
        start = time.time()
        for i, q in enumerate(queries):
            payload = {'id': i, 'text': q, 'max_new_tokens': 120}
            tasks.append(sess.post(url, json=payload))

        responses = await asyncio.gather(*tasks)

        results = []
        for r in responses:
            j = await r.json()
            results.append(j)

        total_time = time.time() - start
        return results, total_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='/nlsasfs/home/ledgerptf/ashsa/models/openhathi/OpenHathi-7B-Hi-v0.1-Base/')
    parser.add_argument('--adapter_dir', default='/nlsasfs/home/ledgerptf/ashsa/models/openhathi/fine_tuned_openhathi/')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--batch_wait_ms', type=int, default=50, help='Batch window in ms')
    parser.add_argument('--max_batch', type=int, default=10)
    args = parser.parse_args()

    print('Starting server process...')
    p = start_server_process(args.model_dir, args.adapter_dir, args.host, args.port, args.batch_wait_ms, args.max_batch)

    # Wait for server to accept connections and report healthy
    def wait_for_server(host: str, port: int, timeout: float = 120.0, interval: float = 0.5):
        import urllib.request
        import urllib.error
        url = f'http://{host}:{port}/health'
        start = time.time()
        while True:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            if time.time() - start > timeout:
                return False
            time.sleep(interval)

    print('Waiting for server to become ready...')
    ready = wait_for_server(args.host, args.port, timeout=300.0, interval=1.0)
    if not ready:
        print('❌ Server did not become ready in time; aborting client requests')
        try:
            p.terminate()
        except Exception:
            pass
        return

    queries = [
        'What is electromagnetic spectrum?',
        'Explain satellite orbits briefly.',
        'What is SAR imaging?',
        'How does atmospheric interference affect remote sensing?',
        'Difference between active and passive remote sensing?',
        'Applications of hyperspectral imaging?',
        'What is geometric correction?',
        'How do microwave sensors work?',
        'Define spatial and spectral resolution.',
        'Advantages of LiDAR technology?'
    ]

    print('Sending concurrent requests...')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results, total_time = loop.run_until_complete(run_client(args.host, args.port, queries))

    print('\nClient finished: total_time=%.3fs' % total_time)
    print('Results:')
    print(json.dumps(results, indent=2))

    try:
        p.terminate()
    except Exception:
        pass


if __name__ == '__main__':
    main()
