import datasets
import torch
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
os.environ["WANDB__SERVICE_WAIT"] = "300"

from collections import namedtuple
import sys
import wandb
import os
import random
import numpy as np
import sae_train

from transformers import AutoTokenizer

from . import cli
from . import data
from .sae_model.configuration_sae import SAEConfig
from .sae_model.modeling_sae import SAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# META CONFIG
parser = cli.setup_parser()
meta_config = parser.parse_args()
gettrace = getattr(sys, 'gettrace', None)
meta_config.is_debug_instance = False if gettrace is None or not gettrace() else True

# SEEDS
meta_config.seed = int(meta_config.seed)
random.seed(meta_config.seed)
np.random.seed(meta_config.seed)
torch.manual_seed(meta_config.seed)

# EXPERIMENT CONFIG
default_experiment_config = {
	**meta_config.__dict__,
}
ExperimentConfig = namedtuple('ExperimentConfig', default_experiment_config.keys())
experiment_config = ExperimentConfig(**default_experiment_config)

# INITIALIZE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# DATA FORMATION
if meta_config.action == 'train-sae' or meta_config.action == 'test-sae':
	dataset = data.load_dataset(meta_config, tokenizer)
else:
	if meta_config.action == 'build_wikipedia_sentences':
		wikipedia = datasets.load_dataset("wikipedia", "20220301.en", split="train", cache_dir=meta_config.input_path)
		data.build_sentence_dataset(meta_config, wikipedia, "wikipedia")
	elif meta_config.action == 'build_bookcorpusopen_sentences':
		bookcorpusopen = datasets.load_dataset("bookcorpusopen", split="train", cache_dir=meta_config.input_path)
		data.build_sentence_dataset(meta_config, bookcorpusopen, "bookcorpusopen")
	elif meta_config.action == 'compute_sentence_statistics':
		dataset = data.load_sentence_dataset(meta_config, meta_config.source_file)
		data.compute_sentence_statistics(meta_config, dataset, tokenizer)
	elif meta_config.action == 'encapsulate_sentence_dataset':
		data.encapsulate_sentence_dataset(meta_config, meta_config.source_file)
	elif meta_config.action == 'combine_sentence_datasets':
		data.combine_sentence_datasets(meta_config)
	elif meta_config.action == 'build_splits':
		data.build_splits(meta_config)
	elif meta_config.action == 'build_sentence_piece_dataset':
		source_dataset = data.form_source_dataset(meta_config)
		data.build_sentence_piece_dataset(meta_config, source_dataset, tokenizer)
	elif meta_config.action == 'build_piece_dataset':
		source_dataset = data.form_source_dataset(meta_config)
		data.build_piece_dataset(meta_config, source_dataset, tokenizer)
	else:
		raise Exception("Unknown action: " + meta_config.action)
	sys.exit(0)

wandb_logging_dir_path = os.path.join(meta_config.output_path, "wandb")
if not os.path.exists(wandb_logging_dir_path):
	os.makedirs(wandb_logging_dir_path)

wandb_project_choice = meta_config.action.split('-')[1]

# INITIALIZE WANDB
experiment_name = f"job-id:{meta_config.job_id}"
wandb.init(
	project=wandb_project_choice+("-proto" if meta_config.is_debug_instance else ""),
	name=experiment_name,
	tags=[
		"job_id:"+str(meta_config.job_id)
	],
	settings=wandb.Settings(start_method='thread'),
	dir=wandb_logging_dir_path,
	config=dict(experiment_config._asdict()) if type(experiment_config).__name__ == 'ExperimentConfig' else dict(experiment_config._as_dict())
)

# RUN
if meta_config.action == 'train-sae' or meta_config.action == 'test-sae':
	config = SAEConfig(
		dim=meta_config.embedding_dim,
		wlt_encoder_num_attention_heads=meta_config.transformer_heads,
		wlt_decoder_num_attention_heads=meta_config.transformer_heads,
		wlt_embedding_width=meta_config.embedding_width,
		wlt_embedding_multiplier=meta_config.embedding_multiplier,
		wlt_encoder_num_hidden_layers=meta_config.transformer_depth,
		wlt_decoder_num_hidden_layers=meta_config.transformer_depth
	)
	if meta_config.action == 'test-sae':
		model_path = os.path.join(meta_config.output_path, meta_config.checkpoint)
		model = torch.load(model_path, map_location=torch.device('cpu'))
	else:
		model = SAE(config)

	# PRINT THE NUMBER OF PARAMETERS
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)
	
	model = model.to(device)
	sae_train.run(meta_config.action, meta_config, experiment_config, dataset, model, tokenizer)
