from typing import Tuple, Any, Optional, List, Dict, Union

import argparse
import ctypes
import tensorrt as trt
import torch
import torch.nn as nn
from diffusers import QwenImageTransformer2DModel
from diffusers.pipelines import QwenImagePipeline
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class QwenImageTRTModel(QwenImageTransformer2DModel):
    dtype_mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.FP8: torch.float8_e4m3fn,
    }
    dtype_mapping_reverse = {
        torch.float32: trt.DataType.FLOAT,
        torch.float16: trt.DataType.HALF,
        torch.bfloat16: trt.DataType.BF16,
        torch.int8: trt.DataType.INT8,
        torch.int32: trt.DataType.INT32,
        torch.int64: trt.DataType.INT64,
        torch.bool: trt.DataType.BOOL,
        torch.uint8: trt.DataType.UINT8,
        torch.float8_e4m3fn: trt.DataType.FP8,
    }

    def __init__(
            self,
            engine_file: str,
            device: torch.device,
            dtype: torch.dtype = torch.bfloat16,
            plugin_file: Optional[str] = None,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        self.plugin_handle = None if plugin_file is None else ctypes.cdll.LoadLibrary(plugin_file)
        torch.cuda.set_device(device)
        trt_logger: trt.Logger = trt.Logger(trt.Logger.VERBOSE)
        assert trt.init_libnvinfer_plugins(trt_logger, '')

        runtime: trt.Runtime
        with open(engine_file, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.STATIC)
        self.engine: trt.ICudaEngine = engine
        self.context: trt.IExecutionContext = context
        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device=device)
        self._device = device
        self._dtype = dtype
        self.stream = torch.cuda.Stream(device)
        config = lambda: None
        config.in_channels = 64
        config.guidance_embeds = False
        self._config = config
        self.register_parameter(
            'fake_empty_tensor',
            nn.Parameter(
                torch.empty(0, dtype=dtype, device=device),
                requires_grad=False,
            )
        )

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @torch.inference_mode()
    def forward(
            self,
            hidden_states: torch.Tensor,  # required
            encoder_hidden_states: torch.Tensor = None,  # required
            encoder_hidden_states_mask: torch.Tensor = None,  # must be None
            timestep: torch.Tensor = None,  # required
            img_shapes: Optional[List[Tuple[int, int, int]]] = None,  # required
            txt_seq_lens: Optional[List[int]] = None,  # required
            guidance: torch.Tensor = None,  # must be None
            attention_kwargs: Optional[Dict[str, Any]] = None,  # must be None
            controlnet_block_samples=None,  # must be None
            return_dict: bool = True,  # all right
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        assert hidden_states.device == encoder_hidden_states.device == timestep.device == self.device
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        timestep = timestep.contiguous()

        rope_complex, encoder_rope_comple = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        hidden_states = hidden_states.to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('hidden_states')],
        )
        timestep = timestep.to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('timestep')],
        )
        encoder_hidden_states = encoder_hidden_states.to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('encoder_hidden_states')],
        )
        img_rope_real = rope_complex.real.contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('img_rope_real')],
            device=self.device,
        )
        img_rope_imag = rope_complex.imag.contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('img_rope_imag')],
            device=self.device,
        )
        txt_rope_real = encoder_rope_comple.real.contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('txt_rope_real')],
            device=self.device,
        )
        txt_rope_imag = encoder_rope_comple.imag.contiguous().to(
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('txt_rope_imag')],
            device=self.device,
        )

        assert self.context.set_tensor_address('hidden_states', hidden_states.data_ptr())
        assert self.context.set_input_shape('hidden_states', tuple(hidden_states.shape))
        assert self.context.set_tensor_address('timestep', timestep.data_ptr())
        assert self.context.set_input_shape('timestep', tuple(timestep.shape))
        assert self.context.set_tensor_address('encoder_hidden_states', encoder_hidden_states.data_ptr())
        assert self.context.set_input_shape('encoder_hidden_states', tuple(encoder_hidden_states.shape))
        assert self.context.set_tensor_address('img_rope_real', img_rope_real.data_ptr())
        assert self.context.set_input_shape('img_rope_real', tuple(img_rope_real.shape))
        assert self.context.set_tensor_address('img_rope_imag', img_rope_imag.data_ptr())
        assert self.context.set_input_shape('img_rope_imag', tuple(img_rope_imag.shape))
        assert self.context.set_tensor_address('txt_rope_real', txt_rope_real.data_ptr())
        assert self.context.set_input_shape('txt_rope_real', tuple(txt_rope_real.shape))
        assert self.context.set_tensor_address('txt_rope_imag', txt_rope_imag.data_ptr())
        assert self.context.set_input_shape('txt_rope_imag', tuple(txt_rope_imag.shape))
        assert self.context.all_shape_inputs_specified and self.context.all_binding_shapes_specified

        out_hidden_states = torch.empty(
            tuple(self.context.get_tensor_shape('out_hidden_states')),
            dtype=self.dtype_mapping[self.engine.get_tensor_dtype('out_hidden_states')],
            device=self.device,
        )
        assert self.context.set_tensor_address('out_hidden_states', out_hidden_states.data_ptr())
        assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        if not return_dict:
            return (out_hidden_states,)

        return Transformer2DModelOutput(sample=out_hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen-Image model path or hf model id")
    parser.add_argument("--trt_path", type=str, required=True,
                        help="trt engine path of Qwen-Image\'s dit")
    parser.add_argument("--plugin_path", type=str, default=None,
                        help="trt engine path of Qwen-Image\'s dit")
    return parser.parse_args()


@torch.inference_mode()
def main(args: argparse.Namespace):
    dtype = torch.bfloat16
    device = torch.device('cuda:0')

    transformer: QwenImageTRTModel = QwenImageTRTModel(
        args.trt_path,
        device=device,
        dtype=dtype,
        plugin_file=args.plugin_path,
    )

    pipe = QwenImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        transformer=None,
    )
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.transformer = transformer

    pipe.to(device=device)
    pipe.set_progress_bar_config(disable=False)

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图."  # for chinese prompt
    }

    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

    negative_prompt = " "  # using an empty string if you do not have specific concept to remove

    width, height = aspect_ratios["16:9"]

    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

    image.save("example.png")


if __name__ == '__main__':
    main(parse_args())
