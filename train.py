import numpy as np
import torch
from loguru import logger
from torch import optim
from utils import gen_mask,KLLoss,split_dataset
from torch.optim import lr_scheduler
from U3T import U3T
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--act_dim', type=int, default=1, help='Action dimension')
parser.add_argument('--context_length', type=int, default=77, help='Context length')
parser.add_argument('--obs_dim', type=int, default=147, help='Observation dimension')
parser.add_argument('--obs_emb_dim', type=int, default=64, help='Observation embedding dimension')
parser.add_argument('--vocab_size', type=int, default=49408, help='Vocabulary size')
parser.add_argument('--trajectory_length', type=int, default=200, help='Trajectory length')
parser.add_argument('--transformer_width', type=int, default=512, help='Transformer width')
parser.add_argument('--transformer_heads', type=int, default=8, help='Transformer heads')
parser.add_argument('--transformer_layers', type=int, default=12, help='Transformer layers')
parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=194, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--dataset', type=str, default="./dataset/data.pkl")

args = parser.parse_args()

if __name__ == '__main__':
    # Create a SummaryWriter for logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir=f'./result/{current_time}/log/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trajectory_length = args.trajectory_length
    context_length = args.context_length
    model=U3T(
        embed_dim = args.embed_dim,
        trajectory_length = args.trajectory_length,
        context_length = args.context_length,
        vocab_size = args.vocab_size,
        transformer_width = args.transformer_width,
        transformer_heads = args.transformer_heads,
        transformer_layers = args.transformer_layers,
        act_dim = args.act_dim,
        obs_dim = args.obs_dim,
        obs_emb_dim = args.obs_emb_dim,
        BERT_PATH='./bert-base-uncased',
        device = device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,betas=(0.9,0.98),eps=1e-8,weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

    loss_trajectory=KLLoss()
    loss_text=KLLoss()
    total_step=0
    curr_total_loss=0
    curr_auc=0
    curr_TTA_loss=0
    curr_CA_loss=0
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    trainset,testset=split_dataset(args.dataset)
    dataloader_train=torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8,collate_fn=lambda x:x)
    dataloader_test=torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=8,collate_fn=lambda x:x)
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader_train, 0):
            model.train()
            transposed_data = list(zip(*data)) 
            obss = transposed_data[0]
            lengths = np.array(transposed_data[3])
            padded_obss = []
            for obs in obss:
                padded_obs = np.pad(obs, ((0, trajectory_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
                padded_obss.append(padded_obs)
            padded_obss = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(device, non_blocking=True)
            acts = transposed_data[1]
            padded_acts=[]
            padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, trajectory_length - len(act)), 'constant', constant_values=(-1)) for act in acts]
            acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(device, non_blocking=True)
            TLss = list(transposed_data[2])
            unique_TLs, mask,count=gen_mask(TLss)
            observations = padded_obss.to(device, non_blocking=True)
            NLss=list(transposed_data[4])
            mask=torch.tensor(mask, device=device, dtype=torch.float)
            input_ids = []
            attention_masks = []
            for sent in NLss:
                encoded_dict=tokenizer.encode_plus(sent, add_special_tokens=True, max_length=context_length, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)
            attention_masks = torch.cat(attention_masks, dim=0).to(device, non_blocking=True)
            optimizer.zero_grad()
            logits_per_trajectory,CA_loss=model(observations,acts,input_ids,attention_masks,lengths)
            TTA_loss=(loss_trajectory(logits_per_trajectory,mask)+loss_text(logits_per_trajectory.t(),mask.t()))/2
            loss = TTA_loss + CA_loss
            loss.backward()
            optimizer.step()
            curr_total_loss+=loss.item()
            curr_TTA_loss+=TTA_loss.item()
            curr_CA_loss+=CA_loss.item()
            if total_step % 10 == 0:
                mask_cpu=mask.cpu()
                logits_per_trajectory_cpu=logits_per_trajectory.cpu().detach()
                roc_auc = roc_auc_score(mask_cpu.flatten(), logits_per_trajectory_cpu.flatten())
                avg_loss = curr_total_loss / 10
                avg_TTA_loss = curr_TTA_loss/10
                avg_CA_loss = curr_CA_loss/10
                logger.info(f'Epoch: {epoch}, Loss: {avg_loss}, TTA_loss:{avg_TTA_loss}, CA_loss:{avg_CA_loss}, AUC: {roc_auc}')
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], total_step)
                writer.add_scalar('Loss/Train_total', avg_loss, total_step)
                writer.add_scalar('Loss/Train_CA', avg_CA_loss, total_step)
                writer.add_scalar('Loss/Train_TTA', avg_TTA_loss, total_step)
                writer.add_scalar('AUC', roc_auc, total_step)
                curr_total_loss, curr_auc,curr_TTA_loss,curr_CA_loss= 0, 0, 0, 0
            total_step+=1
            
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_roc_auc=0.0
            test_step=0
            test_TTA_loss=0
            test_CA_loss=0
            for i, data in enumerate(dataloader_test, 0):
                transposed_data = list(zip(*data)) 
                obss = transposed_data[0]
                lengths = np.array(transposed_data[3])
                padded_obss = []
                for obs in obss:
                    padded_obs = np.pad(obs, ((0, trajectory_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
                    padded_obss.append(padded_obs)
                padded_obss = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(device, non_blocking=True)
                acts = transposed_data[1]
                padded_acts=[]
                padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, trajectory_length - len(act)), 'constant', constant_values=(-1)) for act in acts]
                acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(device, non_blocking=True)
                TLss = list(transposed_data[2])
                unique_TLs, mask,count=gen_mask(TLss)
                observations = padded_obss.to(device, non_blocking=True)
                NLss=list(transposed_data[4])
                mask=torch.tensor(mask, device=device, dtype=torch.float)
                input_ids = []
                attention_masks = []
                for sent in NLss:
                    encoded_dict=tokenizer.encode_plus(sent, add_special_tokens=True, max_length=context_length, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
                    input_ids.append(encoded_dict['input_ids'])
                    attention_masks.append(encoded_dict['attention_mask'])
                input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)
                attention_masks = torch.cat(attention_masks, dim=0).to(device, non_blocking=True)
                logits_per_trajectory,CA_loss=model(observations,acts,input_ids,attention_masks,lengths)
                TTA_loss=(loss_trajectory(logits_per_trajectory,mask)+loss_text(logits_per_trajectory.t(),mask.t()))/2
                mask_cpu=mask.cpu()
                logits_per_trajectory_cpu=logits_per_trajectory.cpu().detach()
                test_roc_auc+=roc_auc_score(mask_cpu.flatten(), logits_per_trajectory_cpu.flatten())
                test_step+=1
                test_loss += loss.item()
                test_CA_loss+=CA_loss.item()
                test_TTA_loss+=TTA_loss.item() 
            writer.add_scalar('Test_learning_rate',optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Loss/Test_all', test_loss/test_step, epoch)
            writer.add_scalar('Loss/Test_CA', test_CA_loss/test_step, epoch)
            writer.add_scalar('Loss/Test_TTA', test_TTA_loss/test_step, epoch)
            writer.add_scalar('Test_AUC', test_roc_auc/test_step, epoch)
        scheduler.step()
        checkpoint_dir = f'./result/{current_time}/model/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = f'./result/{current_time}/model/checkpoint_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path) 
    writer.close()