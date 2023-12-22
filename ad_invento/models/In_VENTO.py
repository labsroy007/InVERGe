import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ad_invento.common.registry import registry
from ad_invento.models.blip2 import Blip2Base, disabled_train
from ad_invento.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from ad_invento.models.mymodel import my_model
import gc
import copy
import torch.nn.functional as F
import torch.distributed as dist
from ad_invento.models.base_model import all_gather_with_grad, concat_all_gather
from ad_invento.models.blip2_outputs import BlipOutput, BlipOutputFeatures

from transformers import AutoTokenizer, TFBertTokenizer

@registry.register_model("In_VENTO")
class INVENTO(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/invento.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        CMQFL_model = "C://Users//Admin//Documents//CVNLP//img2text//MiniGPT-4//checkpoints//blip2_pretrained_flant5xxl.pth", 
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_cmqfl=True,
        freeze_llama=False,
        freeze_Gblip=True,
        train_cmqfl = True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        queue_size=7400,
        momentum=0.9,
        temp=0.9,
        k_test=2000,
        mlm_probability=0.15,
        alpha=0.4
        
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder = my_model.vision_model
        self.visual_encoder_proj = my_model.vision_model.projection_layer
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        print('Loading CMQFL')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 1024  # self.visual_encoder.num_features = 1024
        )
        
        if freeze_cmqfl:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze CMQFL")
        print('Loading CMQFL Done')

                
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        

        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float32,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float32
            )
        if freeze_llama :
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False  
        print('Loading LLAMA Done')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.temp = nn.Parameter(0.07 * torch.ones([])).cpu()
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.train_cmqfl = train_cmqfl
        ###########
#         text_width = self.Qformer.config.hidden_size
#         self.temp = nn.Parameter(torch.ones([]) * temp )   
#         self.queue_size = queue_size
#         self.momentum = momentum
#         self.mlm_probability = mlm_probability
#         self.itm_head = nn.Linear(text_width, 2)  
        
#          # create momentum models
#         self.visual_encoder_m = copy.deepcopy( self.visual_encoder )
#         self.visual_encoder_proj_m = copy.deepcopy( self.visual_encoder_proj )
#         self.Qformer_m = copy.deepcopy(self.Qformer)
        
# #         self.llama_proj_m = copy.deepcopy(self.llama_proj)  
        
#         self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
#                             [self.visual_encoder_proj, self.visual_encoder_proj_m ],
#                             [self.Qformer, self.Qformer_m],  # [self.llama_proj, self.llama_proj_m ],
                            
#                            ]
        
#         self.copy_params()

#         # create the queue
#         self.register_buffer("image_queue", torch.randn(1024, self.queue_size))  # embed_dim image proj
#         self.register_buffer("text_queue", torch.randn(1024, self.queue_size))   # embed_dim OF text proj
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
#         self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
#         self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        
        
        ###########
        
        
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, targets=None, input_ids=None):
        torch.cuda.empty_cache()
        gc.collect()
        device = image.device   # cuda:0 --> AD
        
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        
        with self.maybe_autocast():
            torch.cuda.empty_cache()
            gc.collect()
            image_embeds = self.visual_encoder(image)
            image_embeds = self.visual_encoder_proj(image_embeds)#.to(device)
            
            

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)#.to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            self.Qformer.bert = self.Qformer.bert.to('cpu')  # AD
            query_tokens = query_tokens.to('cpu')  # AD
            
            if targets==None:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,  # encoder_embeds
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True
                )
            else:
                query_output = self.Qformer(
                    input_ids=input_ids,
                    query_embeds=query_tokens,  # encoder_embeds
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=False,
                    labels = targets
                )
            inputs_llama = self.llama_proj(query_output.last_hidden_state.to('cuda') )
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
            
        torch.cuda.empty_cache()
        gc.collect()
        
        image = samples["image"]
        text = samples["text_input"]
        
        ##########
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        
        with self.maybe_autocast():
            torch.cuda.empty_cache()
            gc.collect()
            image_embeds = self.visual_encoder(image)
            image_embeds = self.visual_encoder_proj(image_embeds)#.to(device)
#             image_feat = F.normalize(torch.mean(image_embeds, dim=1) , dim=-1) # Assume image_embeds[:,0,:]==torch.mean(image_embeds, dim=1)
            
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)#.to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        self.Qformer.bert = self.Qformer.bert.to('cpu')  # AD
        query_tokens = query_tokens.to('cpu')  # AD

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,  # encoder_embeds
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True, 
            use_cache=True,
        )

        image_feats = F.normalize(
            query_output.last_hidden_state, dim=-1
        )
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            text_output.last_hidden_state[:, 0, :], dim=-1
        )
        if self.train_cmqfl :
            ###============== Image-text Contrastive ===================###
            image_feats_all = concat_all_gather(
                image_feats
            )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
            text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

            sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            sim_i2t = sim_i2t.cpu() / self.temp.cpu()

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i.cpu() / self.temp.cpu()  # [batch_size, batch_size*num_gpu]
            try:
                rank = dist.get_rank()  
            except :
                rank = 1  # <- For me no distribute system
            bs = image.size(0)
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
                image.device
            )

            if "image_id" in samples.keys(): #coco retrieval finetuning
                image_ids = samples["image_id"].view(-1,1)
                image_ids_all = concat_all_gather(image_ids)
                pos_idx = torch.eq(image_ids, image_ids_all.t()).float()       
                sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
                sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)
                sim_targets = sim_targets.cpu()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
                loss_itc = (loss_t2i+loss_i2t)/2  
            else:                     
                loss_itc = (
                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                ) / 2

            ###============== Image-text Matching ===================###
            text_input_ids_world = concat_all_gather(text_tokens.input_ids)
            text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
            image_embeds_world = all_gather_with_grad(image_embeds)
            with torch.no_grad():
                if "image_id" in samples.keys():
                    mask = torch.eq(image_ids, image_ids_all.t()).cpu()
                    sim_t2i.masked_fill_(mask, -10000)
                    sim_i2t.masked_fill_(mask, -10000)
                else:    
                    sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                    sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)

            # select a negative image for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1).cpu()
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1).cpu()

            image_embeds_all = torch.cat(
                [image_embeds, image_embeds_neg, image_embeds], dim=0
            )  # pos, neg, pos
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).cpu()

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
            )
            self.itm_head.cpu()
            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :].cpu()
            vl_output = self.itm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(image.device)
            loss_itm = F.cross_entropy(logits, itm_labels)

            ##================= Image Captioning ========================##
            decoder_input_ids = text_tokens.input_ids.clone()
    #         decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss 
            print("loss_mlm", loss_lm, "loss_ita", loss_itc, "loss_itm", loss_itm )
            return self.train_cmqfl, {"loss_mlm": loss_lm, "loss_ita": loss_itc, "loss_itm": loss_itm }
        
        else: 
            #================= Time of fine-tune the decoder ========================##


            inputs_llama = self.llama_proj(query_output.last_hidden_state.to('cuda') )
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

            img_embeds, atts_img = inputs_llama, atts_llama  # self.encode_img(image)  # , targets, bert_tokens.input_ids 

            if hasattr(samples, 'question_split'):  # VQA dataset
                print('VQA Batch')
                vqa_prompt = '###Human: <Img><ImageHere></Img> '
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
            elif self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                           dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=to_regress_tokens.input_ids.dtype,
                             device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            atts_bos=atts_bos.to('cuda')
            atts_img = atts_img.to('cuda')

            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask.to('cuda')], dim=1)
            torch.cuda.empty_cache()
            gc.collect()
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            print("loss_mlm", loss)
            return self.train_cmqfl, {"loss_mlm": loss}
    
    
        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data.cpu()
                param.data = param.data.cpu()
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        CMQFL_model = cfg.get("CMQFL_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_cmqfl = cfg.get("freeze_cmqfl", True)
        freeze_llama = cfg.get("freeze_llama", True)
        train_cmqfl = cfg.get("train_cmqfl", True)
        freeze_Gblip = cfg.get("freeze_Gblip", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        ####
        queue_size = cfg.get("queue_size")
        momentum = cfg.get("momentum", 0.9)
        temp = cfg.get("temp", 0.9)
        k_test = cfg.get("k_test", 2000)
        mlm_probability = cfg.get("mlm_probability", 0.15)
        alpha = cfg.get("alpha", 0.4)
        
        ####
        model = cls(
            vit_model=vit_model,
            CMQFL_model=CMQFL_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_cmqfl=freeze_cmqfl,
            freeze_llama=freeze_llama,
            train_cmqfl=train_cmqfl,
            freeze_Gblip=freeze_Gblip,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            queue_size=queue_size,
            momentum=momentum,
            temp=temp,
            k_test=k_test,
            mlm_probability=mlm_probability,
            alpha=alpha
        )
        
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            keys_to_remove = ['llama_proj.weight', 'llama_proj.bias']

            for key in keys_to_remove:
                if key in ckpt['model']:
                    del ckpt['model'][key]

            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model


        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
#     tensors_gather = [torch.ones_like(tensor)  for _ in range(1)]  # torch.distributed.get_world_size() =1
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
    return tensor
