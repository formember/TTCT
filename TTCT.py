from typing import Tuple, Union
from torch import nn
from model import Transformer, LayerNorm
import torch
from transformers import BertModel,BertTokenizer
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TTCT(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 act_dim: int,
                 obs_dim,
                 obs_emb_dim,
                 trajectory_length: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 BERT_PATH,
                 device,
                 threshold=None,
                 episodic_cost_value=None
                 ):
        super().__init__()
        self.threshold=threshold
        self.episodic_cost_value=episodic_cost_value
        self.device=device
        self.embed_dim=embed_dim
        self.obs_dim=obs_dim
        self.context_length = context_length
        self.trajectory_length=trajectory_length
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_emb_dim),
        )
        self.trajectory_inner_loss=nn.CrossEntropyLoss()
        self.trajectory_transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_trajectory_attention_mask()
        )
        self.text_model = BertModel.from_pretrained(BERT_PATH)
        self.cost_assignment_layer = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        self.episodic_cost_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.error=torch.nn.MSELoss()
        self.embedding_act = nn.Linear(act_dim, 16)
        self.obs_encoder_linear = nn.Linear(obs_emb_dim, transformer_width-16)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.transformer_width=transformer_width
        self.trajectory_positional_embedding = nn.Parameter(torch.empty(self.trajectory_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.trajectory_ln_final=LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(768, embed_dim))
        self.trajectory_projection=nn.Parameter(torch.empty(transformer_width,embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.embedding_act.weight, std=0.02)
        nn.init.normal_(self.obs_encoder_linear.weight, std=0.02)
        nn.init.normal_(self.trajectory_positional_embedding, std=0.01)
        nn.init.orthogonal_(self.cost_assignment_layer[0].weight)
        nn.init.orthogonal_(self.cost_assignment_layer[2].weight)
        nn.init.orthogonal_(self.episodic_cost_layer[0].weight)
        nn.init.orthogonal_(self.episodic_cost_layer[2].weight)
        proj_std = (self.trajectory_transformer.width ** -0.5) * ((2 * self.trajectory_transformer.layers) ** -0.5)
        attn_std = self.trajectory_transformer.width ** -0.5
        fc_std = (2 * self.trajectory_transformer.width) ** -0.5
        for block in self.trajectory_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=768** -0.5)
        
        if self.trajectory_projection is not None:
            nn.init.normal_(self.trajectory_projection, std=self.trajectory_transformer.width ** -0.5)

    def build_trajectory_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.trajectory_length, self.trajectory_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    @property
    def dtype(self):
        return torch.float32

    def encode_observation(self, image):#[batch_size, trajectory_len, image_resolution]
        return self.obs_encoder(image.type(self.dtype))
    
    
    def regression(self,trajector, text):
        x = torch.cat([trajector, text.unsqueeze(0).repeat(trajector.shape[-2],1)], dim=-1)
        return self.cost_assignment_layer(x)
    
    def encode_trajectory(self, trajectory,lengths,text_featrues):# [batch_size, trajectory_len, d_model]
        x = trajectory
        # [batch_size, trajectory_len, vision_dim]
        x = x + self.trajectory_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trajectory_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.trajectory_ln_final(x).type(self.dtype)
        x = x @ self.trajectory_projection
        x = x / x.norm(dim=-1, keepdim=True)
        
        
        text_embed=text_featrues
        logit_scale = self.logit_scale.exp()
        cos_sim = torch.matmul(logit_scale * x , text_embed.unsqueeze(2)).squeeze(-1)
        cos_sim = cos_sim.masked_fill(torch.tensor(lengths, dtype=torch.int32).view(-1, 1).to(self.device) 
                                      <= torch.arange(cos_sim.size(1)).to(self.device), float('-inf'))
        atten_score = F.sigmoid(cos_sim)
        hidden_embed = atten_score.unsqueeze(2) * x
        cost_assignment_loss = 0
        episodic_cost = self.episodic_cost_layer(text_embed.detach())
        for i in range(hidden_embed.size(0)):
            single_cost=self.regression(hidden_embed[i, :lengths[i]-1, :].detach(),text_embed[i,:].detach())
            sum_cost=torch.sum(single_cost)
            cost_assignment_loss += (self.error(sum_cost,episodic_cost[i][0])+self.error(episodic_cost[i][0],sum_cost))/2
        cost_assignment_loss = cost_assignment_loss / hidden_embed.size(0)
        cost_assignment_loss += self.trajectory_inner_loss(cos_sim,torch.tensor([item-1 for item in lengths]).to(self.device))
        last_embed = torch.stack([x[i, lengths[i]-1, :] for i in range(x.size(0))]) 
        return last_embed, cost_assignment_loss
    
    def test_encode_text(self,text):
        input_ids = []
        attention_masks = []
        for sent in text:
            encoded_dict=self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=77, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device, non_blocking=True)
        attention_masks = torch.cat(attention_masks, dim=0).to(self.device, non_blocking=True)
        text_features=self.encode_text(input_ids, attention_masks)
        return text_features
    
    
    def test_encode_trajectory(self, trajectory,length,lengths):
        x = trajectory
        #[batch_size, trajectory_len, vision_dim]
        x = x + self.trajectory_positional_embedding.type(self.dtype)[0:length,:]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trajectory_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.trajectory_ln_final(x).type(self.dtype)
        x = x @ self.trajectory_projection
        
        #[batch, d_model]
        last_embed = torch.stack([x[i, lengths[i]-1, :] for i in range(x.size(0))]) 
        
        return last_embed
    
    
    def test_encode(self,trajectory,actions,lengths,text_features):
        batchsize=len(trajectory)
        max_length=max(lengths)
        padded_obss = []
        last_obs=[]
        for obs in trajectory:
            last_obs.append(obs[-1])
            padded_obs = np.pad(obs, ((0, max_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
            padded_obss.append(padded_obs)
        last_obs=torch.tensor(np.array(last_obs), dtype=torch.float32).to(self.device)
        last_obs=last_obs.view(batchsize,-1)
        trajectory = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(self.device)
        
        padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, max_length - len(act)), 'constant', constant_values=(-1)) for act in actions]
        padded_acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(self.device, non_blocking=True)
        actions_view = padded_acts.view(batchsize,max_length,-1)
        actions_emb=self.embedding_act(actions_view)
        action_features=actions_emb.view(batchsize,max_length,-1)
        
        trajectory=trajectory.view(batchsize,max_length,-1)
        trajectory_image_features = self.encode_observation(trajectory)
        trajectory_image_features=self.obs_encoder_linear(trajectory_image_features)
        trajectory_image_features = torch.cat([trajectory_image_features, action_features], dim=-1)
            
        last_embed = self.test_encode_trajectory(trajectory_image_features,max_length,lengths)
        return torch.cat((last_obs,last_embed,text_features),dim=-1)
    
    
    def encode_text(self, input_ids, attention_mask):
        output_attentions = False
        output_hidden_states = False
        return_dict = True
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = text_outputs[1]
        text_features = x @ self.text_projection
        return text_features
    
    
    def get_cost_per_trajectory(self,trajectory,text_featrues,length,is_predict_cost):
        x = trajectory
        #[batch_size, trajectory_len, vision_dim]
        x = x + self.trajectory_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trajectory_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.trajectory_ln_final(x).type(self.dtype)
        x = x @ self.trajectory_projection
        
        embed=x[0, 0:length, :]
        embed_norm=embed/embed.norm(dim=-1, keepdim=True)
        text_featrues_norm = text_featrues / text_featrues.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        scores = torch.matmul(logit_scale * embed_norm.unsqueeze(1) , text_featrues_norm).squeeze()
        final_cost = scores > self.threshold
        if is_predict_cost:
            atten_score = F.sigmoid(scores)
            embed_norm = atten_score * embed_norm
            predict_cost = self.regression(embed_norm,text_featrues_norm).squeeze()
            final_cost = torch.where(final_cost == 0, predict_cost, self.episodic_cost_value)
        else:
            final_cost = torch.where(final_cost == 0, 0, self.episodic_cost_value)
        return final_cost
        
        
    
    def get_cost(self,trajectory,actions,text_features,is_predict_cost):
        length=len(trajectory[0])
        padded_obss = []
        for obs in trajectory:
            padded_obs = np.pad(obs, ((0, self.trajectory_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
            padded_obss.append(padded_obs)
        trajectory = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(self.device)

        padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, self.trajectory_length - len(act)), 'constant', constant_values=(-1)) for act in actions]
        padded_acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(self.device)
        actions_view = padded_acts.view(1,self.trajectory_length,-1)
        actions_emb=self.embedding_act(actions_view)
        action_features=actions_emb.view(1,self.trajectory_length,-1)

        trajectory=trajectory.view(1,self.trajectory_length,-1)
        trajectory_image_features = self.encode_observation(trajectory)
        trajectory_image_features=self.obs_encoder_linear(trajectory_image_features)
        trajectory_image_features = torch.cat([trajectory_image_features, action_features], dim=-1)
        return self.get_cost_per_trajectory(trajectory_image_features,text_features,length,is_predict_cost)
    
    
    
    def forward(self, observations, actions, input_ids, attention_mask, lengths):
        batch_size=observations.shape[0]
        actions=actions.view(batch_size*self.trajectory_length,-1)
        actions=self.embedding_act(actions)
        action_features=actions.view(batch_size,self.trajectory_length,-1)
        observations=torch.flatten(observations, start_dim=0, end_dim=1)
        observations=observations.view(batch_size,self.trajectory_length,-1)
        observation_features = self.encode_observation(observations)
        
        observation_features=observation_features.view(batch_size,self.trajectory_length,-1)
        observation_features=self.obs_encoder_linear(observation_features)
        #[batch_size, trajectory_len, d_model] -> [batch_size, d_model]
        
        # 2*[batch_size, trajectory_len, d_model//2] -> [batch_size, trajectory_len, d_model]
        trajectory_features = torch.cat([observation_features, action_features], dim=-1)
        #[text_batch_size, n_ctx, d_model] -> [text_batch_size, d_model]
        text_features = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)  
        
        # normalized features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        trajectory_features, cost_assignment_loss = self.encode_trajectory(trajectory_features,lengths,text_features)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        #[batch_size, text_batch_size]
        logits_per_trajectory = logit_scale * trajectory_features @ text_features.t()
            
        return logits_per_trajectory, cost_assignment_loss
    

