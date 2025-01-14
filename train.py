import os
import logging
from datetime import datetime
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from modules.stage1_model import ImageToSoftPrompt
from modules.stage2_model import SoftPromptImageCaption
from data import get_dataloader


def train_stage1(args, model, accelerator):
    train_dataloader, val_dataloader = get_dataloader(args, model.tokenizer, model.processor, seq_length=args.seq_length, mode='stage1')
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad is True], lr=args.lr_stage1, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epoch_stage1 * len(train_dataloader))
    best_loss = 1024

    # set logger
    if accelerator.is_main_process and args.log_train:
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(args.log_dir, 'train_log_stage1_' + timestamp + '.log')
        logging.basicConfig(filename=file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    for i in range(args.n_epoch_stage1):
        avg_loss = 0.0
        iter_loss = 0.0
        model.train()

        if accelerator.is_main_process:
            train_dataloader = tqdm(train_dataloader, desc=f"Epoch {i + 1}", leave=True, dynamic_ncols=True)

        for step, batch in enumerate(train_dataloader):
            align_losses, aux_losses = 0.0, 0.0
            with accelerator.accumulate(model):
                imgs, captions = batch
                for j in range(captions.shape[1]):
                    align_loss, aux_loss = model(imgs, caption_token_ids=captions[:, j], calibration=True)
                    align_losses += align_loss
                    aux_losses += aux_loss
                
                loss = align_losses / captions.shape[1] + 2.0 * aux_losses / captions.shape[1]
   
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += loss.item()
            avg_loss += loss.item()

            if accelerator.is_main_process and (step + 1) % args.log_iter == 0:
                accelerator.print(f"Epoch {i + 1} - Loss: {iter_loss / args.log_iter}")
                if args.log_train:
                    logger.info(f"Epoch {i + 1} - Loss: {iter_loss / args.log_iter}")
                iter_loss = 0.0
        
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {i + 1} Finish, Average Loss: {avg_loss / len(train_dataloader)}")
            if args.log_train:
                logger.info(f"Epoch {i + 1} Finish, Average Loss: {avg_loss / len(train_dataloader)}")

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for step, batch in enumerate(tqdm(val_dataloader)):
                    imgs, captions = batch

                    align_losses, aux_losses = 0.0, 0.0
                    for j in range(captions.shape[1]):
                        align_loss, aux_loss = model(imgs, caption_token_ids=captions[:, j], calibration=True)
                        align_losses += align_loss
                        aux_losses += aux_loss

                    loss = align_losses / captions.shape[1] + 2.0 * aux_losses / captions.shape[1]
                    val_loss += loss.item()

                accelerator.print(f"Average Validation Loss: {val_loss / len(val_dataloader)}")
                if args.log_train:
                    logger.info(f"Average Validation Loss: {val_loss / len(val_dataloader)}")

            if val_loss / len(val_dataloader) <= best_loss:
                model.save(os.path.join(args.output_dir, 'generator_best.pth'))
                accelerator.print(f"Best model saved to {args.output_dir}")
                best_loss = val_loss / len(val_dataloader)
            
            accelerator.wait_for_everyone()
            

def train_stage2(args, model, accelerator):
    train_dataloader, val_dataloader = get_dataloader(args, model.tokenizer, model.processor, seq_length=args.seq_length, mode='stage2')
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad is True], lr=args.lr_stage2, weight_decay=args.weight_decay)
    best_loss = 1024
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epoch_stage2 * len(train_dataloader))

    # set logger
    if accelerator.is_main_process and args.log_train:
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(args.log_dir, 'train_log_stage2_' + timestamp + '.log')
        logging.basicConfig(filename=file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    for i in range(args.n_epoch_stage2):
        avg_loss = 0.0
        iter_loss = 0.0
        model.train()

        if accelerator.is_main_process:
            train_dataloader = tqdm(train_dataloader, desc=f"Epoch {i + 1}", leave=True, dynamic_ncols=True)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                imgs, captions = batch

                loss = 0.0
                for j in range(captions.shape[1]):
                    caption_loss = model(imgs, captions[:, j])
                    loss += caption_loss

                loss = loss / captions.shape[1]

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += loss.item()
            avg_loss += loss.item()

            if accelerator.is_main_process and (step + 1) % args.log_iter == 0:
                accelerator.print(f"Iteration {(step + 1)} - Loss: {iter_loss / args.log_iter}")
                if args.log_train:
                    logger.info(f"Iteration {(step + 1)} - Loss: {iter_loss / args.log_iter}")
                iter_loss = 0.0
        
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {i + 1} Finish, Average Loss: {avg_loss / len(train_dataloader)}")
            if args.log_train:
                logger.info(f"Epoch {i + 1} Finish, Average Loss: {avg_loss / len(train_dataloader)}")
            avg_loss = 0.0

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for step, batch in enumerate(tqdm(val_dataloader)):
                    imgs, captions = batch

                    loss = 0.0
                    for j in range(captions.shape[1]):
                        caption_loss = model(imgs, captions[:, j])
                        loss += caption_loss
                    
                    loss = loss / captions.shape[1]
                    val_loss += caption_loss.item()

                accelerator.print(f"Average Validation Loss: {val_loss / len(val_dataloader)}")
                if args.log_train:
                    logger.info(f"Average Validation Loss: {val_loss / len(val_dataloader)}")
            
            if val_loss / len(val_dataloader) <= best_loss:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_best.pth'))
                accelerator.print(f"Best model saved to {args.output_dir}")
                best_loss = val_loss / len(val_dataloader)
            
            accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="Arguments for pruning mask training")
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
        '--seed', type=int, default=42,
        help='random seed'
    )
    parser.add_argument(
        '--lr_stage1', type=float, default=1e-5,
        help='learning rate for stage1 training'
    )
    parser.add_argument(
        '--lr_stage2', type=float, default=1e-5,
        help='learning rate for stage2 training'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=5e-2,
        help='learning rate for training mask'
    )
    parser.add_argument(
        '--n_epoch_stage1', type=int, default=20,
        help='number of epochs for training mask'
    )
    parser.add_argument(
        '--n_epoch_stage2', type=int, default=10,
        help='number of epochs for training mask'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size for training mask'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='gradient accumulation steps during training'
    )
    parser.add_argument(
        '--stage1_ckpt', type=str, default=None,
        help='ckpt for stage2 training, skipping stage1'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='output path for saving model'
    )
    parser.add_argument(
        '--log_train', action='store_true', default=False,
        help='log result during training'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./log',
        help='output log path'
    )
    parser.add_argument(
        '--log_iter', type=int, default=100,
        help='logging interval during training'
    )

    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator.print(f"Running on device: {accelerator.device}")

    stage1_model = ImageToSoftPrompt(
        encoder_model_name=args.encoder_model_name,
        base_model_name=args.base_model_name,
        prompt_len=args.prompt_len,
        num_layers=args.num_layers,
    )

    if args.stage1_ckpt is None:
        train_stage1(args, stage1_model, accelerator)
        stage1_model.load(os.path.join(args.output_dir, 'generator_best.pth'))
    else:
        stage1_model.load(args.stage1_ckpt)
        accelerator.print(f"Loaded stage1 model from {args.stage1_ckpt}")

    stage2_model = SoftPromptImageCaption(
        image_to_soft_prompt_model=stage1_model,
        base_model_name=args.base_model_name,
        device=accelerator.device,
    )

    torch.cuda.empty_cache()

    train_stage2(args, stage2_model, accelerator)


if __name__ == '__main__':
    main()