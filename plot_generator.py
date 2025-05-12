import torch
import os
import sys
import json
import numpy as np
sys.path.append('/workspace/SteerKep/activation-steering')
sys.path.append("/workspace/SteerKep/SteerPoser/src")
sys.path.append("/workspace/SteerKep/steer-data")
sys.path.append("/workspace/SteerKep/RLBench")

from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, MalleableModel, SteeringVector
from activation_steering.steering_vector import *
from arguments import get_config
from steered_model import *
steer_cfg = get_config(config_path='/workspace/SteerKep/steer-data/steerconfig.yaml')

print("Finished imports and got steer config")

model = AutoModelForCausalLM.from_pretrained(steer_cfg.model_name, cache_dir=steer_cfg.cache_dir, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(steer_cfg.model_name, cache_dir=steer_cfg.cache_dir)

## Load original 

print("Loaded model and tok")

def load_dataset(ds_name, tokenizer, chat_temp=False):
    with open(os.path.join(steer_cfg.steering_datasets_dir, ds_name), 'r') as f:
        dset = json.load(f)
    examples = []
    suffixes = []
    for item in dset:
        examples.append((item["input"], item["input"]))
        suffixes.append((item["compliant_continuation"], item["non_compliant_continuation"]))
    return SteeringDataset(tokenizer, examples, suffixes, use_chat_template=chat_temp) 


def load_sv(sv_name):
    svpath = os.path.join(steer_cfg.steering_vector_dir, sv_name)
    return SteeringVector.load(svpath)

sv_ds_name = "junk-healthy.json"
sv_dataset = load_dataset(sv_ds_name, tokenizer)

print("Loaded steering dataset ", sv_dataset, "\n")

steer_vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=sv_dataset,
    method="pca_center",
    accumulate_last_x_tokens="suffix-only",
    save_analysis=True,
    output_dir="/workspace/SteerKep/activation_steering_figures/train_healthy/"
)

steer_vector.save(os.path.join(steer_cfg.steering_vector_dir, "junk-healthy2.svec"))

print(f"Trained and saved steering vector to {os.path.join(steer_cfg.steering_vector_dir, 'junk-healthy2.svec')}")

with open("/workspace/SteerKep/saved_results.json", "r") as file:
    saved_results = json.load(file)
sammy_hidden = saved_results["sammy"]["hidden"]
dan_hidden = saved_results["danelle"]["hidden"]
aakash_hidden = saved_results["aakash"]["hidden"]

sammy_hidden = {int(k): np.array(v) for k, v in sammy_hidden.items()}
dan_hidden = {int(k): np.array(v) for k, v in dan_hidden.items()}
aakash_hidden = {int(k): np.array(v) for k, v in aakash_hidden.items()}

hidden_layer_ids=[29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]


print("opened and saved hidden states from interactions. Now creating strength plots")

save_sv_strength_figures(sammy_hidden, hidden_layer_ids, "/workspace/SteerKep/activation_steering_figures/sammy_proj/", steer_vector)

save_sv_strength_figures(dan_hidden, hidden_layer_ids, "/workspace/SteerKep/activation_steering_figures/danelle_proj/", steer_vector)

save_sv_strength_figures(aakash_hidden, hidden_layer_ids, "/workspace/SteerKep/activation_steering_figures/aakash_proj/", steer_vector)











