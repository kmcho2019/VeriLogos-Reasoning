#!/bin/bash

accelerate launch --config_file {config_file} main.py RLTF -im deepseek-ai/deepseek-coder-6.7b-instruct -om VeriLogos_RLTF -d rltf.jsonl


