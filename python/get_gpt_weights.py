# Load model directly
import json
from collections import OrderedDict
from tqdm import tqdm
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Prepare the data structure for JSON serialization
weights_dict = OrderedDict()

# Iterate over the model parameters and store their weights
for name, param in tqdm(model.named_parameters(), total=148):
    print(name, param.shape)
    weights_dict[name.replace("transformer.", "")] = {
        "shape": list(param.shape),
        "data": param.flatten().data.cpu().numpy().tolist()  # Convert to list
    }

# Serialize to JSON with indentation
with open('model_weights.json', 'w') as f:
    json.dump(weights_dict, f, indent=4)
