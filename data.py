import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class Flickr30kDataset(Dataset):
    def __init__(self, root='./datas/Flickr30k', tokenizer=None, processor=None, max_length=40, split='train'):
        self.root = root
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.processor = processor
        self.max_length = max_length

        self.train_dataset_path = os.path.join(self.root, 'train')
        self.val_dataset_path = os.path.join(self.root, 'val')
        self.test_dataset_path = os.path.join(self.root, 'test')

        self.train_imgs_path = os.listdir(os.path.join(self.train_dataset_path, 'train_img'))
        self.train_captions_path = os.path.join(self.train_dataset_path, 'train.token')
        with open(self.train_captions_path, 'r', encoding='utf-8') as file:
            self.train_captions = file.readlines()

        self.val_imgs_path = os.listdir(os.path.join(self.val_dataset_path, 'val_img'))
        self.val_captions_path = os.path.join(self.val_dataset_path, 'val.token')
        with open(self.val_captions_path, 'r', encoding='utf-8') as file:
            self.val_captions = file.readlines()

        self.test_imgs_path = os.listdir(os.path.join(self.test_dataset_path, 'test_img'))

        self.split = split

    def load_paired_data(self, img_name):
        caption_token_ids = []
        cations = self.train_captions if self.split == 'train' else self.val_captions
        for caption in cations:
            if img_name in caption:
                caption = caption.split('\t')[1].strip()
                caption_tokens = self.tokenizer.encode(caption, max_length=self.max_length, truncation=True, padding='max_length', add_special_tokens=False)
                caption_token_ids.append(torch.tensor(caption_tokens, dtype=torch.int64))

        return torch.stack(caption_token_ids)

    def __len__(self):
        if self.split == 'train':
            return len(self.train_imgs_path)
        elif self.split == 'val':
            return len(self.val_imgs_path)
        else:
            return len(self.test_imgs_path)

    def __getitem__(self, idx):
        if self.split == 'train':
            img_name = self.train_imgs_path[idx]
            img_path = os.path.join(os.path.join(self.train_dataset_path, 'train_img'), img_name)
        elif self.split == 'val':
            img_name = self.val_imgs_path[idx]
            img_path = os.path.join(os.path.join(self.val_dataset_path, 'val_img'), img_name)
        else:
            img_name = self.test_imgs_path[idx]
            img_path = os.path.join(os.path.join(self.test_dataset_path, 'test_img'), img_name)

        img = Image.open(img_path).convert('RGB')
        img = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)

        if self.split == 'train' or self.split == 'val':
            caption = self.load_paired_data(img_name)
            return img, caption
        else:
            return img_name, img


def get_dataloader(args, tokenizer, processor, seq_length, mode='stage1'):
    if mode == 'stage1' or mode == 'stage2':
        train_dataset = Flickr30kDataset(tokenizer=tokenizer, processor=processor, max_length=seq_length, split='train')
        val_dataset = Flickr30kDataset(tokenizer=tokenizer, processor=processor, max_length=seq_length, split='val')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        return train_dataloader, val_dataloader
    elif mode == 'test':
        test_dataset = Flickr30kDataset(tokenizer=tokenizer, processor=processor, max_length=seq_length, split='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        return test_dataloader


if __name__ == "__main__":
    from transformers import AutoTokenizer, CLIPProcessor

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = Flickr30kDataset(tokenizer=tokenizer, processor=processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for img, caption in dataloader:
        print(img.keys())
        print(caption.shape)
        break