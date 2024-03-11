import torch
from transformers import AutoModelForCausalLM

from modeling_phi2 import PhiForCausalLM
from config_phi2 import PhiConfig

def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)

def load():
    model = AutoModelForCausalLM.from_pretrained("if001/sample_phi-2")
    for v in model.state_dict():
        print(v)

def upload():
    config = PhiConfig()    
    config.hidden_size=8
    config.num_attention_heads=4
    config.num_key_value_heads=2
    config.num_hidden_layers=6
    config.intermediate_size = 10
    print(config)
    model = PhiForCausalLM(config)
    model.apply(init_weights)
    print(model)
    
    # from torchinfo import summary
    # summary(model=model)
    model.push_to_hub('if001/sample_phi-2')

def main():
    upload()
    # load()

if __name__ == "__main__":
    main()