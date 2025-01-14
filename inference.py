import torch
import argparse
from accelerate import Accelerator
from modules.stage1_model import ImageToSoftPrompt
from modules.stage2_model import SoftPromptImageCaption
from data import get_dataloader


def inference(args, model, accelerator):
    test_loader = get_dataloader(args, model.tokenizer, model.processor, seq_length=args.seq_length, mode='test')

    model, test_loader = accelerator.prepare(model, test_loader)

    total_captions = {}
    model.eval()
    with torch.inference_mode():
        for img_name, imgs in test_loader:
            captions = model.generate(imgs)
            print(captions)
            # break

            total_captions.update({img_name[i]: captions[i] for i in range(len(img_name))})

    with open('test_captions.txt', 'w') as f:
        for key, value in total_captions.items():
            f.write(f'{key}\t{value}\n')


def main():
    parser = argparse.ArgumentParser(description="")
    # model options
    parser.add_argument(
        '--encoder_model_name', type=str, default='openai/clip-vit-large-patch14',
        help='target model for feature extractions'
    )
    parser.add_argument(
        '--base_model_name', type=str, default='TinyLlama/TinyLlama_v1.1',
        help='target base model for generating captions'
    )
    parser.add_argument(
        '--prompt_len', type=int, default=32,
        help='generated soft prompt length for soft prompt generator'
    )
    parser.add_argument(
        '--num_layers', type=int, default=3,
        help='# of layers for soft prompt generator'
    )
    parser.add_argument(
        '--seq_length', type=int, default=32,
        help='pad length'
    )

    # train options
    parser.add_argument(
        '--ckpt', type=str, default='./output/model_best.pth',
        help=''
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='batch size for training mask'
    )

    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision='bf16', 
    )
    accelerator.print(f"Running on device: {accelerator.device}")

    stage1_model = ImageToSoftPrompt(
        encoder_model_name=args.encoder_model_name,
        base_model_name=args.base_model_name,
        prompt_len=args.prompt_len,
        num_layers=args.num_layers,
    )

    stage2_model = SoftPromptImageCaption(
        image_to_soft_prompt_model=stage1_model,
        base_model_name=args.base_model_name,
        device=accelerator.device,
    )

    stage2_model.load_state_dict(torch.load(args.ckpt))
    accelerator.print(f"Loaded model from {args.ckpt}")
    stage2_model = stage2_model.to(dtype=torch.bfloat16)

    torch.cuda.empty_cache()

    inference(args, stage2_model, accelerator)


if __name__ == '__main__':
    main()