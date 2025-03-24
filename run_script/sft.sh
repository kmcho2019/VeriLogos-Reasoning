#!/bin/bash

accelerate launch --config_file {config_file} main.py SFT -im deepseek-ai/deepseek-coder-6.7b-instruct -om VeriLogos -d sft.jsonl
