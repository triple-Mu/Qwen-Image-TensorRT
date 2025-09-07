#!/bin/bash

TENSORRT_ROOT="TensorRT-10.13.2.6"
ONNX_PATH="transformer_step2.onnx"
ENGINE_PATH="transformer_step2.engine"

export PATH=${TENSORRT_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${TENSORRT_ROOT}/lib:$LD_LIBRARY_PATH

which trtexec

trtexec \
  --onnx=${ONNX_PATH} \
  --saveEngine=${ENGINE_PATH} \
  --bf16 \
  --optShapes=hidden_states:1x6032x64,encoder_hidden_states:1x128x3584,timestep:1,img_rope_real:6032x64,img_rope_imag:6032x64,txt_rope_real:128x64,txt_rope_imag:128x64 \
  --minShapes=hidden_states:1x3364x64,encoder_hidden_states:1x1x3584,timestep:1,img_rope_real:3364x64,img_rope_imag:3364x64,txt_rope_real:1x64,txt_rope_imag:1x64 \
  --maxShapes=hidden_states:1x10816x64,encoder_hidden_states:1x1024x3584,timestep:1,img_rope_real:10816x64,img_rope_imag:10816x64,txt_rope_real:1024x64,txt_rope_imag:1024x64 \
  --shapes=hidden_states:1x10816x64,encoder_hidden_states:1x1024x3584,timestep:1,img_rope_real:10816x64,img_rope_imag:10816x64,txt_rope_real:1024x64,txt_rope_imag:1024x64 \
  2>&1 | tee ${ONNX_PATH}.log

