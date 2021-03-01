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

    
    def forward(self, input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, mode):
        if mode == "train":
            (lm_loss), (mc_loss), *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, lm_labels=lm_labels
            )

            return lm_loss, mc_loss
        
        if mode=="eval":
            lm_logits, mc_logits, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            
            print('lm_logits is ', lm_logits)
            print('mc_logits is ', mc_logits)
            
            return lm_logits, mc_logits
