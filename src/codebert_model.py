import torch
from torch import nn
from transformers import AutoModel

# set up the neural network for CodeBERT fine-tuning
class BERT_Arch(nn.Module):
    def __init__(self, auto_model, output_layer):
        super(BERT_Arch, self).__init__()
        self.bert = auto_model
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU() #For activation
        #self.fc1 = nn.Linear(2*768, 512)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, output_layer)
        self.softmax = nn.LogSoftmax(dim=-1)

         # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True


    # define the forward pass
    def forward(self, sent_id, mask, is_embedding=False):
        if is_embedding:
            # If sent_id is actually embeddings
            embeddings = sent_id
            chunk_size = 512  # For chunking
            total_seq_length = embeddings.size(1)
            cls_hs = []
            for start in range(0, total_seq_length, chunk_size):
                end = min(start + chunk_size, total_seq_length)
                chunk_embeddings = embeddings[:, start:end, :]
                outputs = self.bert(inputs_embeds=chunk_embeddings, attention_mask=mask)
                cls_hs_current = outputs[0][:, 0, :]
                cls_hs.append(cls_hs_current)
            cls_hs = torch.cat(cls_hs, dim=1)
        else:
            sent_id = sent_id.long()
            mask = mask.long()
            chunk_size = 512  # For chunking
            total_seq_length = sent_id.size(1)
            cls_hs = []
            for start in range(0, total_seq_length, chunk_size):
                end = min(start + chunk_size, total_seq_length)
                chunk_sent_id = sent_id[:, start:end]
                chunk_mask = mask[:, start:end]
		    	
		    	# pass the inputs to the model
                #cls_hs_current = self.bert(chunk_sent_id, attention_mask=chunk_mask)[1] #This calls the original codebert model
                outputs = self.bert(chunk_sent_id, attention_mask=chunk_mask)
                cls_hs_current = outputs[1] #[:, 0, :] #outputs[1] 
                cls_hs.append(cls_hs_current)
                #print(f'Chunk {start} to {end}, cls_hs_current shape: {cls_hs_current.shape}')  # Debugging line

            cls_hs = torch.cat(cls_hs, dim=1)
        fc1_output = self.fc1(cls_hs)
        relu_output = self.relu(fc1_output)
        dropout_output = self.dropout(relu_output)
        fc2_output = self.fc2(dropout_output)
        final_output = self.softmax(fc2_output)
        return final_output
        #return fc2_output
    
    def get_embeddings(self, input_ids):
        #print('Getting bert embeddings....')
        embeddings = self.bert.embeddings(input_ids)
        return embeddings #.float()
#
#class BERT_Arch(nn.Module):
#    def __init__(self, auto_model, output_layer):
#        super(BERT_Arch, self).__init__()
#        self.bert = auto_model
#        self.dropout = nn.Dropout(0.2)
#        self.relu = nn.ReLU()
#        self.fc1 = nn.Linear(768, 512)
#        self.fc2 = nn.Linear(512, output_layer)
#        self.softmax = nn.LogSoftmax(dim=-1)
#
#    def forward(self, sent_id, mask):
#        sent_id = sent_id.long()
#        mask = mask.long()
#        print(f'Forward method input shapes: sent_id {sent_id.shape}, mask {mask.shape}')  # Debugging line
#        outputs = self.bert(sent_id, attention_mask=mask)
#        #print(f'BERT outputs: {outputs}')
#        cls_hs = outputs[1]  # Pooled output
#        fc1_output = self.fc1(cls_hs)
#        relu_output = self.relu(fc1_output)
#        dropout_output = self.dropout(relu_output)
#        fc2_output = self.fc2(dropout_output)
#        final_output = self.softmax(fc2_output)
#        return final_output

