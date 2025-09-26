import os
import sys
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

if len(sys.argv) < 2:
    print("Usage: python save_bias_model.py <path_to_pruned_model_sd_pt>")
    sys.exit(1)

PT_PATH = sys.argv[1]
BASE_MODEL = "huggyllama/llama-7b"
SAVE_DIR = f"models/{os.path.basename(PT_PATH).replace('.pt', '').replace('.sd', '')}"
num_layers = 32

# load config and modify to include biases
config = LlamaConfig.from_pretrained(BASE_MODEL)
config.attention_bias = True
config.mlp_bias = True
config.use_cache = False

# instantiate fresh model with bias-enabled config
print(config)
print('initializing...')
model_bias_ready = LlamaForCausalLM(config)
# bias to zero
for i in range(num_layers):
    layer = model_bias_ready.model.layers[i]

    # attention
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        module = getattr(layer.self_attn, proj)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

    # mlp
    for proj in ["up_proj", "gate_proj", "down_proj"]:
        module = getattr(layer.mlp, proj)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
print(model_bias_ready)


# load state_dict from pruned model with bias
model_with_bias = torch.load(PT_PATH)
print(model_with_bias)
for mod in model_with_bias:
    print(mod)

# extract weights and biases
model_bias_ready.load_state_dict(model_with_bias, strict=False)
# check
print(model_bias_ready.model.layers[0].self_attn.q_proj.bias)
print(model_bias_ready.model.layers[0].self_attn.o_proj.bias)

# save
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model_bias_ready.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Saved model with bias to: {SAVE_DIR}")
