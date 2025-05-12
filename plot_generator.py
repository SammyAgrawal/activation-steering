import torch
import os
import sys
import json
sys.path.append('/workspace/SteerKep/activation-steering')
sys.path.append("/workspace/SteerKep/SteerPoser/src")
sys.path.append("/workspace/SteerKep/steer-data")
sys.path.append("/workspace/SteerKep/RLBench")

from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, MalleableModel, SteeringVector
from arguments import get_config
from steered_model import *
steer_cfg = get_config(config_path='/workspace/SteerKep/steer-data/steerconfig.yaml')

## Load original 

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

sv_ds_name = "junk-healthy-24b.svec"
sv_dataset = load_sv(sv_name)

ds_name = "sammy_rated_interactions.json"

with open("/workspace/SteerKep/saved_results.json", "r") as file:
    saved_results = json.load(file)


