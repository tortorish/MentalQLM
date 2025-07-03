#!/bin/bash

# specify path of your trained model here
vllm serve autodl-tmp/mentalqlm \
--port 5004 \
--max-model-len 4096 \
--gpu-memory-utilization 0.9 