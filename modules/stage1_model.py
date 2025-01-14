import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoConfig, 
    AutoModel,
    CLIPTokenizer
)
from modules.custom_embed_clip import CustomCLIPTextTransformer


class ImageEncoder(nn.Module):
    def __init__(self, vision_model, visual_projection):
        super(ImageEncoder, self).__init__()

        self.model = vision_model
        self.proj = visual_projection

        self.image_feature_dim = self.model.config.projection_dim

    def forward(self, imgs):
        image_features = self.model(imgs)[1]
        image_features = self.proj(image_features)

        return image_features


class SoftPromptGenerator(nn.Module):
    def __init__(self, model_name, feature_dim, prompt_len, num_layers):
        super(SoftPromptGenerator, self).__init__()

        self.model_name = model_name

        self.prompt_len = prompt_len
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        self.generator_config = AutoConfig.from_pretrained(model_name)
        self.generator_config.num_hidden_layers = self.num_layers
        self.embed_dim = self.generator_config.hidden_size

        self.in_proj = nn.Linear(feature_dim, prompt_len * self.embed_dim)

        self.generator = self.build_generator()
        self.embed_idx = -1
        
        self.output_projection = nn.Linear(self.embed_dim, feature_dim)

        self.init_keywords_dict = {
            'TinyLlama/TinyLlama_v1.1':['self_attn.o_proj', 'mlp.down_proj']
        }

    def build_generator(self, is_zero_init=False):
        new_generator = AutoModel.from_config(self.generator_config)

        # zero out related parameters
        if is_zero_init:
            keyword_list = self.init_keywords_dict[self.model_name]

            for module_name, module in new_generator.named_modules():
                if module_name in keyword_list:
                    assert isinstance(module, nn.Linear), "error for zeroing out"
                    for name, val in module.named_parameters():
                        setattr(module, name, nn.Parameter(torch.zeros_like(val)))
            
            new_generator.wpe.weight = nn.Parameter(torch.zeros_like(new_generator.wpe.weight))

        return new_generator

    def forward(self, image_features):
        projected_features = self.in_proj(image_features).reshape(-1, self.prompt_len, self.embed_dim)

        soft_prompt = self.generator(inputs_embeds=projected_features, output_hidden_states=True)['hidden_states'][self.embed_idx].to(torch.bfloat16)

        soft_prompt = self.output_projection(soft_prompt)

        return soft_prompt 


class PromptCalibration(nn.Module):
    def __init__(self, text_model, text_projection, logit_scale):
        super(PromptCalibration, self).__init__()

        self.model = text_model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.proj = text_projection
        self.logit_scale = logit_scale

        self.embed = self.model.embeddings.token_embedding
        self.eos_token = self.model.config.eos_token_id
        self.register_buffer(
            'eos_token_embed', self.embed.weight[self.eos_token]
        )

        self.text_feature_dim = self.model.config.projection_dim
    
    def forward(self, soft_prompt, image_features, caption_token_ids=None):
        text_features = self.model(inputs_embeds=soft_prompt)[1]
        text_features = self.proj(text_features)
    
        # normalized features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t().to(text_features.device)) * logit_scale.to(text_features.device)

        loss = self.clip_loss(logits_per_text)

        if caption_token_ids is not None:
            label_features = self.model(caption_token_ids)[1]
            label_features = self.proj(label_features)
            label_features = label_features / label_features.norm(p=2, dim=-1, keepdim=True)

            logits = torch.matmul(label_features, text_features.t().to(label_features.device)) * logit_scale.to(label_features.device)
            # logits = torch.matmul(label_features, image_features.t().to(label_features.device)) * logit_scale.to(label_features.device)

            aux_loss = self.clip_loss(logits)

            return loss, aux_loss
        
        return loss
    
    def clip_loss(self, similarity):
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
    
    @staticmethod
    def contrastive_loss(logits):
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


class ImageToSoftPrompt(nn.Module):
    def __init__(self, encoder_model_name, base_model_name, prompt_len, num_layers):
        super(ImageToSoftPrompt, self).__init__()

        full_model = CLIPModel.from_pretrained(
            encoder_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='auto',
        )

        self.processor = CLIPProcessor.from_pretrained(encoder_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(encoder_model_name)

        for param in full_model.parameters():
            param.requires_grad = False

        self.image_encoder = ImageEncoder(
            full_model.vision_model, 
            full_model.visual_projection
        )

        self.soft_prompt_generator = SoftPromptGenerator(
            base_model_name,
            feature_dim=full_model.config.projection_dim,
            prompt_len=prompt_len,
            num_layers=num_layers
        )

        self.prompt_calibration = PromptCalibration(
            self.create_custom_model(full_model.text_model),
            full_model.text_projection,
            full_model.logit_scale,
        )

        self.align_dim = full_model.config.projection_dim
    
    def create_custom_model(self, origin_text_model):
        new_model = CustomCLIPTextTransformer(origin_text_model.config)
        new_model.load_state_dict(origin_text_model.state_dict())

        return new_model

    def forward(self, imgs, caption_token_ids=None, calibration=False):
        image_features = self.image_encoder(imgs)

        soft_prompts = self.soft_prompt_generator(image_features)

        if calibration:
            loss = self.prompt_calibration(soft_prompts, image_features, caption_token_ids)
            return loss
        else:
            return soft_prompts
    
    def save(self, path):
        torch.save(self.soft_prompt_generator.state_dict(), path)
    
    def load(self, path):
        self.soft_prompt_generator.load_state_dict(torch.load(path))