import torch
from torch import nn

from transformers import OpenAIGPTDoubleHeadsModel,GPT2DoubleHeadsModel


ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


class MyModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(MyModel, self).__init__()
        model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
        self.model = model_class.from_pretrained(args.model_checkpoint)
        self.model.to(args.device)
        add_special_tokens_(self.model, tokenizer)
        self.config = self.model.config

    
    def forward(self, input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, 
                emotion_labels, arousal_labels, valence_labels, mode):
        if mode == "train":
            loss, mc_loss, logits, mc_logits, hidden_states, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, lm_labels=lm_labels
            )
            
            hidden_states = hidden_states[-1]
            x = hidden_states[:,-1,-1,:] # (batch, num_candidates, seq_len, hidden_size)
            x = x.reshape(-1, self.input_dims)
            out = self.fc1(x)
            out = self.fc2(out)
            out_e = self.softmax(self.fc3(out))
            out_a = self.tanh_a(self.fc4(out))
            out_v = self.tanh_v(self.fc5(out))
            
            loss_e = self.loss(out_e, emotion_labels)
            loss_a = self.loss_mse(out_a, arousal_labels)
            loss_v = self.loss_mse(out_v, valence_labels
            
            return loss, mc_loss, loss_e, loss_a, loss_v

        if mode=="eval":
            lm_logits, mc_logits, hidden_states, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )

            
            hidden_states = hidden_states[-1]
            x = hidden_states[:,-1,-1,:]
            x = x.reshape(-1, self.input_dims)
            out = self.fc1(x)
            out = self.fc2(out)
                                   
            out_e = self.softmax(self.fc3(out))
            
            out_a = self.tanh_a(self.fc4(out))
            out_v = self.tanh_v(self.fc5(out))
            
            return lm_logits, mc_logits, out_e, out_a, out_v
      
    def generate(self, input_ids, token_type_ids, max_length, min_length=1, temperature=0.7):
        self.model.eval()
        
#         greedy_output = self.model.generate(input_ids, max_length)
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        current_output = []
        
        for i in range(max_length):
            logits,mc_logits, *_ = self.model(input_ids, token_type_ids=token_type_ids) 
            logits = logits[0, -1, :] / temperature
            logits = top_filtering(logits)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if True else torch.multinomial(probs, 1)
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        
        
        return current_output
