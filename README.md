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

*COMING SOON!*