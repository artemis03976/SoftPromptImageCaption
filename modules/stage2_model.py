import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from peft import LoraConfig, get_peft_model


peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)


class DynamicSoftEmbedding(nn.Module):
    def __init__(
            self,
            wte: nn.Embedding,
            bos_token_id,
    ):
        super(DynamicSoftEmbedding, self).__init__()

        self.wte = wte
        self.register_buffer(
            'bos_token_embed', self.wte.weight[bos_token_id]
        )

    def forward(self, tokens, soft_prompts):
        input_embeddings = self.wte(tokens)

        # bos_token_embed = self.bos_token_embed.unsqueeze(0).repeat(tokens.shape[0], 1, 1)

        return torch.cat([soft_prompts, input_embeddings], dim=1)


class BaseModel(nn.Module):
    def __init__(self, model_name="TinyLlama/TinyLlama_v1.1"):
        super(BaseModel, self).__init__()

        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.embedding = DynamicSoftEmbedding(self.model.get_input_embeddings(), self.tokenizer.bos_token_id)

        for param in self.model.parameters():
            param.requires_grad = False
        
        # self.model = get_peft_model(self.model, peft_config)

    def forward(self, soft_prompts, input_ids):
        inputs_embeds = self.embedding(input_ids, soft_prompts)

        return self.model(inputs_embeds=inputs_embeds)
    
    def generate(self, soft_prompts, input_ids, num_return_sequences=5):
        inputs_embeds = self.embedding(input_ids, soft_prompts)

        return self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=32,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2 
        )


class PromptTemplate:
    def __init__(self, tokenizer, device):
        # self.manual_prompt = "This is a picture that shows"
        self.manual_prompt = "The picture clearly depicts that"

        self.tokenizer = tokenizer

        self.manual_prompt_ids = self.tokenizer(self.manual_prompt, return_tensors="pt")["input_ids"].to(device)


class SoftPromptImageCaption(nn.Module):
    def __init__(
            self,
            image_to_soft_prompt_model,
            base_model_name="TinyLlama/TinyLlama_v1.1",
            device='cuda',
    ):
        super(SoftPromptImageCaption, self).__init__()

        self.image_to_soft_prompt_model = image_to_soft_prompt_model

        self.base_model_name = base_model_name
        self.model = BaseModel(self.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.processor = self.image_to_soft_prompt_model.processor

        self.template = PromptTemplate(self.tokenizer, device)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.soft_prompts_proj = nn.Linear(
            self.image_to_soft_prompt_model.align_dim,
            self.model.config.hidden_size
        )

    def forward(self, imgs, caption_token_ids):
        soft_prompts = self.image_to_soft_prompt_model(imgs)
        soft_prompts = self.soft_prompts_proj(soft_prompts)

        # Concatenate the template and the captions
        prompt_template_ids = self.template.manual_prompt_ids.repeat(caption_token_ids.size(0), 1)
        input_ids = torch.cat([prompt_template_ids, caption_token_ids], dim=1)

        # Pass the input_ids and soft_prompts to the model
        output = self.model(soft_prompts, input_ids)
        logits = output.logits
        # only compute loss on the caption part
        logits = logits[:, -input_ids.shape[1]:]

        # mask everything except caption
        labels = torch.clone(input_ids)
        labels[labels == self.tokenizer.pad_token_id] = -100

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        caption_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return caption_loss
    
    def generate(self, imgs, num_return_sequences=5):
        soft_prompts = self.image_to_soft_prompt_model(imgs)
        soft_prompts = self.soft_prompts_proj(soft_prompts)

        # Concatenate the template and the captions
        prompt_template_ids = self.template.manual_prompt_ids.repeat(soft_prompts.size(0), 1)

        results = self.model.generate(soft_prompts, prompt_template_ids, num_return_sequences=num_return_sequences)

        sentences = self.tokenizer.batch_decode(results, skip_special_tokens=True)

        captions = []
        for sentence in sentences:
            sentence = sentence.split('\n')[0]
            captions.append(sentence)

        return captions

