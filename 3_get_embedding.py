import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel

# 输入文件夹是你把DNABERT2放到的同一个文件夹的path
config = BertConfig.from_pretrained("../../DNABERT_2/")
tokenizer = AutoTokenizer.from_pretrained("../../DNABERT_2/", trust_remote_code=True)
dna_model = AutoModel.from_pretrained("../../DNABERT_2/", trust_remote_code=True, config=config)


from tqdm import tqdm
import torch

# 检查是否有可用的GPU
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

batch_size = 16  

with open('./center.fa', 'r') as file:
    lines = file.readlines()

batch_sequences = []

dna_model = dna_model.to(device)

seq_vec = torch.empty((0, dna_model.config.hidden_size), device=device)  # 假设模型的输出维度为hidden_size

with torch.no_grad():
    for line in tqdm(lines, desc='Processing seq'):
        if line[0] == '>':
            continue
        batch_sequences.append(line.strip().upper()) 

        if len(batch_sequences) == batch_size:
            
            inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True)["input_ids"].to(device)  
            hidden_states = dna_model(inputs)[0]  
            
            batch_embeddings = torch.mean(hidden_states, dim=1)  #

            seq_vec = torch.cat((seq_vec, batch_embeddings), dim=0)
            
            batch_sequences = []

    if len(batch_sequences) > 0:
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True)["input_ids"].to(device)
        hidden_states = dna_model(inputs)[0]
        batch_embeddings = torch.mean(hidden_states, dim=1)
        seq_vec = torch.cat((seq_vec, batch_embeddings), dim=0)

seq_vec = seq_vec.to('cpu')
torch.save(seq_vec, 'seq.pth')
