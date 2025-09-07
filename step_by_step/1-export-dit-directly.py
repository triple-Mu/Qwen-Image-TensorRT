import os
import tempfile
import argparse

import onnx
import torch
from diffusers import QwenImageTransformer2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Qwen/Qwen-Image',
                        help="Qwen-Image model path or hf model id")
    parser.add_argument("--onnx_path", type=str, default='transformer.onnx',
                        help="ONNX path of exported Qwen-Image\'s dit")
    return parser.parse_args()


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

    out_hidden_states = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=None,
        timestep=timestep,
        img_shapes=[[(batch_size, img_height // 16, img_width // 16)]],
        txt_seq_lens=[txt_seq_len],
        guidance=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        return_dict=False,
    )[0]

    print(f'{out_hidden_states.shape}\n', end='')

    with tempfile.TemporaryDirectory() as d:
        temp_path = f'{d}/{os.path.basename(args.onnx_path)}'
        torch.onnx.export(
            transformer,
            (
                hidden_states,  # hidden_states
                encoder_hidden_states,  # encoder_hidden_states
                None,  # encoder_hidden_states_mask
                timestep,  # timestep
                [[(batch_size, img_height // 16, img_width // 16)]],  # img_shapes
                [txt_seq_len],  # txt_seq_lens
                None,  # guidance
                None,  # attention_kwargs
                None,  # controlnet_block_samples
                False,  # return_dict
            ),
            temp_path,
            opset_version=17,
            input_names=[
                'hidden_states',
                'encoder_hidden_states',
                'encoder_hidden_states_mask',
                'timestep',
                'img_shapes',
                'txt_seq_lens',
                'guidance',
                'attention_kwargs',
                'controlnet_block_samples',
                'return_dict',
            ],
            output_names=['out_hidden_states'],
            dynamic_axes={
                'hidden_states': {1: 'img_seq_len'},
                'encoder_hidden_states': {1: 'txt_seq_len'},
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
