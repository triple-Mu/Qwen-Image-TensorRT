# Qwen-Image-TensorRT

Qwen-Image's DiT inference with TensorRT-10

## ENV

The project was tested in the following environment:

- Ubuntu 18.04
- NVIDIA Driver 525.125.06
- [`CUDA`](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run)
  11.8
- Python 3.10.18
- [
  `PyTorch`](https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=715d3b039a629881f263c40d1fb65edac6786da13bfba221b353ef2371c4da86)
  2.6.0+cu118
- [`Diffusers`](https://github.com/huggingface/diffusers/commit/fc337d585309c4b032e8d0180bea683007219df1) 0.36.0.dev0
- [
  `ONNX`](https://files.pythonhosted.org/packages/79/21/9bcc715ea6d9aab3f6c583bfc59504a14777e39e0591030e7345f4e40315/onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl)
  1.19.0
- [
  `TensorRT`](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/tars/TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz)
  10.13.0.35
- [`cudnn-frontend`](https://github.com/NVIDIA/cudnn-frontend/commit/1a7b4b78db44712fb9707d21cd2e3179f1fd88b8) 1.14.1

```shell
# Create conda env
conda create -n qwen-image python=3.10
conda activate qwen-image

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# Install Diffusers
pip install git+https://github.com/huggingface/diffusers.git@fc337d585309c4b032e8d0180bea683007219df1
# Install ONNX
pip install onnx==1.19.0

# Install TensorRT
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/tars/TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz
tar -xf TensorRT-10.13.0.35.Linux.x86_64-gnu.cuda-11.8.tar.gz
pip install TensorRT-10.13.2.6/python/tensorrt-10.13.2.6-cp310-none-linux_x86_64.whl
export PATH=${PWD}/TensorRT-10.13.2.6/bin:$PATH
export LD_LIBRARY_PATH=${PWD}/TensorRT-10.13.2.6/lib:$LD_LIBRARY_PATH

# Install cudnn-frontend
# tensorrt-plugin is coming soon
```

## CONVERT TO ONNX

Clone the project first:

```shell
git clone https://github.com/triple-Mu/Qwen-Image-TensorRT.git
cd Qwen-Image-TensorRT
```

Here are some scripts to test exporting onnx:

- [`1-export-dit-directly.py`](./step_by_step/1-export-dit-directly.py)

```shell
python step_by_step/1-export-dit-directly.py --model_path Qwen/Qwen-Image --onnx_path transformer_step1.onnx
```

This script almost no modifications, so the export fails with the following error:

```text
  File "/root/anaconda3/envs/qwen-image/lib/python3.10/site-packages/torch/onnx/_internal/jit_utils.py", line 308, in _create_node
    _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
RuntimeError: ScalarType ComplexFloat is an unexpected tensor scalar type
```

Since ONNX does not support complex operators, proceed to step 2.

- [`2-remove-complex-op.py`](./step_by_step/2-remove-complex-op.py)

```shell
python step_by_step/2-remove-complex-op.py --model_path Qwen/Qwen-Image --onnx_path transformer_step2.onnx
```

After removing `self.pos_embed` and replacing `apply_rotary_emb_qwen`, it works fine.

- [`3-merge-qkv-projection.py`](./step_by_step/3-merge-qkv-projection.py)

```shell
python step_by_step/3-merge-qkv-projection.py --model_path Qwen/Qwen-Image --onnx_path transformer_step3.onnx
```

Advanced: Merging QKV GEMM reduces kernel launches and increases throughput.

- [`4-cudnn-attention-plugin.py`](./step_by_step/4-cudnn-attention-plugin.py)

```shell
python step_by_step/4-cudnn-attention-plugin.py --model_path Qwen/Qwen-Image --onnx_path transformer_step4.onnx
```

*COMING SOON!*

Advanced: Replacing sdpa with cudnn-attention, it results in a significant improvement on A100 GPU.

## CONVERT TO TensorRT

After convert `QwenImageTransformer2DModel` to ONNX, the tensorrt engine can be built by `trtexec`.

Refer to [`2-build_engine.sh`](./scripts/2-build_engine.sh)

Set up `TENSORRT_ROOT` `ONNX_PATH` and `ENGINE_PATH` first, and the min/opt/max shape also can be modified it yourself.

Then run:

```shell
bash scripts/2-build_engine.sh
```

The following log output will be shown:

```text
[09/07/2025-21:42:26] [I] === Trace details ===
[09/07/2025-21:42:26] [I] Trace averages of 10 runs:
[09/07/2025-21:42:26] [I] Average on 10 runs - GPU latency: 1666.2 ms - Host latency: 1666.9 ms (enqueue 1663.95 ms)
[09/07/2025-21:42:26] [I] 
[09/07/2025-21:42:26] [I] === Performance summary ===
[09/07/2025-21:42:26] [I] Throughput: 0.562059 qps
[09/07/2025-21:42:26] [I] Latency: min = 1656.22 ms, max = 1674.64 ms, mean = 1666.9 ms, median = 1667.89 ms, percentile(90%) = 1673.26 ms, percentile(95%) = 1674.64 ms, percentile(99%) = 1674.64 ms
[09/07/2025-21:42:26] [I] Enqueue Time: min = 1650.99 ms, max = 1672.49 ms, mean = 1663.95 ms, median = 1663.63 ms, percentile(90%) = 1672.08 ms, percentile(95%) = 1672.49 ms, percentile(99%) = 1672.49 ms
[09/07/2025-21:42:26] [I] H2D Latency: min = 0.631348 ms, max = 0.640015 ms, mean = 0.635217 ms, median = 0.635742 ms, percentile(90%) = 0.63623 ms, percentile(95%) = 0.640015 ms, percentile(99%) = 0.640015 ms
[09/07/2025-21:42:26] [I] GPU Compute Time: min = 1655.52 ms, max = 1673.94 ms, mean = 1666.2 ms, median = 1667.19 ms, percentile(90%) = 1672.56 ms, percentile(95%) = 1673.94 ms, percentile(99%) = 1673.94 ms
[09/07/2025-21:42:26] [I] D2H Latency: min = 0.0585938 ms, max = 0.0664062 ms, mean = 0.0639648 ms, median = 0.0644531 ms, percentile(90%) = 0.0654297 ms, percentile(95%) = 0.0664062 ms, percentile(99%) = 0.0664062 ms
[09/07/2025-21:42:26] [I] Total Host Walltime: 17.7917 s
[09/07/2025-21:42:26] [I] Total GPU Compute Time: 16.662 s
[09/07/2025-21:42:26] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[09/07/2025-21:42:26] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[09/07/2025-21:42:26] [I] Explanations of the performance metrics are printed in the verbose logs.
[09/07/2025-21:42:26] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v101300] [b35] # trtexec --onnx=transformer_step2.onnx --saveEngine=transformer_step2.plan --bf16 --optShapes=hidden_states:1x6032x64,encoder_hidden_states:1x128x3584,timestep:1,img_rope_real:6032x64,img_rope_imag:6032x64,txt_rope_real:128x64,txt_rope_imag:128x64 --minShapes=hidden_states:1x3364x64,encoder_hidden_states:1x1x3584,timestep:1,img_rope_real:3364x64,img_rope_imag:3364x64,txt_rope_real:1x64,txt_rope_imag:1x64 --maxShapes=hidden_states:1x10816x64,encoder_hidden_states:1x1024x3584,timestep:1,img_rope_real:10816x64,img_rope_imag:10816x64,txt_rope_real:1024x64,txt_rope_imag:1024x64 --shapes=hidden_states:1x10816x64,encoder_hidden_states:1x1024x3584,timestep:1,img_rope_real:10816x64,img_rope_imag:10816x64,txt_rope_real:1024x64,txt_rope_imag:1024x64
```

## RUNNING TensorRT Pipeline!

After convert ONNX to Engine, the pipeline can be built with Diffusers's pipeline.

Refer to [`run_trt_pipeline.py`](./run_trt_pipeline.py)

Run:

```shell
python run_trt_pipeline.py --model_path Qwen/Qwen-Image --trt_path transformer_step2.engine
```

Then the example output image will be saved at [`example.png`](./example.png).

## CUDNN-ATTENTION Plugin!

*COMING SOON!*