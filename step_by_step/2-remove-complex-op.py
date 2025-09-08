from typing import Tuple
import os
import tempfile
import argparse

import onnx
import torch
import torch.nn as nn
import diffusers
import diffusers.models.transformers.transformer_qwenimage
from diffusers import QwenImageTransformer2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Qwen/Qwen-Image',
                        help="Qwen-Image model path or hf model id")
    parser.add_argument("--onnx_path", type=str, default='transformer.onnx',
                        help="ONNX path of exported Qwen-Image\'s dit")
    return parser.parse_args()


def apply_rotary_emb_qwen(
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
) -> torch.Tensor:
    x_fp32 = x.float()  # improve precision
    x_fp32 = x_fp32.unflatten(-1, (-1, 2))
    a, b = x_fp32.unbind(-1)  # [b, s, n, d//2]
    c, d = freqs_cis  # [s, d//2]
    c = c[None, :, None, :]  # [1, s, 1, d//2]
    d = d[None, :, None, :]  # [1, s, 1, d//2]
    real = (a * c - b * d).to(x.dtype)
    imag = (b * c + a * d).to(x.dtype)
    y = torch.stack([real, imag], dim=-1)
    return y.flatten(-2)


diffusers.models.transformers.transformer_qwenimage.apply_rotary_emb_qwen = apply_rotary_emb_qwen


class OptDit(nn.Module):
    def __init__(self, dit: QwenImageTransformer2DModel):
        super().__init__()
        self.dit = dit

    def forward(
            self,
            hidden_states: torch.Tensor,  # bf16 [batch, img_seq_len, 64]
            encoder_hidden_states: torch.Tensor,  # bf16 [batch, txt_seq_len, 3584]
            timestep: torch.Tensor,  # bf16 [batch]
            img_rope_real: torch.Tensor,  # fp32 [img_seq_len, 128//2]
            img_rope_imag: torch.Tensor,  # fp32 [img_seq_len, 128//2]
            txt_rope_real: torch.Tensor,  # fp32 [txt_seq_len, 128//2]
            txt_rope_imag: torch.Tensor,  # fp32 [txt_seq_len, 128//2]
    ) -> torch.Tensor:
        hidden_states = self.dit.img_in(hidden_states)
        encoder_hidden_states = self.dit.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.dit.txt_in(encoder_hidden_states)
        temb = self.dit.time_text_embed(timestep, hidden_states)

        for block in self.dit.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=((img_rope_real, img_rope_imag), (txt_rope_real, txt_rope_imag)),
                joint_attention_kwargs={},
            )
        hidden_states = self.dit.norm_out(hidden_states, temb)
        output = self.dit.proj_out(hidden_states)
        return output


@torch.inference_mode()
def main(args: argparse.Namespace):
    dtype = torch.bfloat16
    device = torch.device('cuda:0')

    transformer: QwenImageTransformer2DModel = QwenImageTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder='transformer',
        torch_dtype=dtype,
    )

    transformer.eval()
    transformer.to(dtype=dtype, device=device)

    img_width = 1664
    img_height = 928

    batch_size = 1
    img_seq_len = img_width // 16 * img_height // 16
    txt_seq_len = 256
    in_channels = transformer.config.in_channels  # 64
    joint_attention_dim = transformer.config.joint_attention_dim  # 3584
    attention_head_dim = transformer.config.attention_head_dim  # 128

    hidden_states = torch.randn(
        (batch_size, img_seq_len, in_channels),
        dtype=dtype,
        device=device,
    )
    timestep = torch.randint(
        0, 1000,
        (batch_size,),
        device=device,
    ).to(dtype=dtype)
    encoder_hidden_states = torch.randn(
        (batch_size, txt_seq_len, joint_attention_dim),
        dtype=dtype,
        device=device,
    )
    img_rope_real = torch.randn(
        (img_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    img_rope_imag = torch.randn(
        (img_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    txt_rope_real = torch.randn(
        (txt_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )
    txt_rope_imag = torch.randn(
        (txt_seq_len, attention_head_dim // 2),
        dtype=torch.float32,
        device=device,
    )

    transformer: OptDit = OptDit(transformer)

    out_hidden_states = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_rope_real=img_rope_real,
        img_rope_imag=img_rope_imag,
        txt_rope_real=txt_rope_real,
        txt_rope_imag=txt_rope_imag,
    )

    print(f'{out_hidden_states.shape}\n', end='')

    with tempfile.TemporaryDirectory() as d:
        temp_path = f'{d}/{os.path.basename(args.onnx_path)}'
        torch.onnx.export(
            transformer,
            (
                hidden_states,  # hidden_states
                encoder_hidden_states,  # encoder_hidden_states
                timestep,  # timestep
                img_rope_real,  # img_rope_real
                img_rope_imag,  # img_rope_imag
                txt_rope_real,  # txt_rope_real
                txt_rope_imag,  # txt_rope_imag
            ),
            temp_path,
            opset_version=17,
            input_names=[
                'hidden_states',
                'encoder_hidden_states',
                'timestep',
                'img_rope_real',
                'img_rope_imag',
                'txt_rope_real',
                'txt_rope_imag',
            ],
            output_names=['out_hidden_states'],
            dynamic_axes={
                'hidden_states': {1: 'img_seq_len'},
                'encoder_hidden_states': {1: 'txt_seq_len'},
                'img_rope_real': {0: 'img_seq_len'},
                'img_rope_imag': {0: 'img_seq_len'},
                'txt_rope_real': {0: 'txt_seq_len'},
                'txt_rope_imag': {0: 'txt_seq_len'},
                'out_hidden_states': {1: 'img_seq_len'},
            }
        )
        onnx_model = onnx.load(temp_path)
        onnx.save(
            onnx_model,
            args.onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=args.onnx_path.replace('.onnx', '.onnx.data'),
        )


if __name__ == '__main__':
    main(parse_args())
