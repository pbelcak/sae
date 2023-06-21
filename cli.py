import argparse
import time

def setup_parser():
	# meta config zone
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-j',
		'--job-id',
		type=int,
		default=int(time.time()),
		help='The job id (and the name of the wandb group)'
	)
	parser.add_argument(
		'-o',
		'--output-path',
		type=str,
		default="models",
		help='The directory which will contain saved model checkpoints'
	)

	parser.add_argument(
		'-i',
		'--input-path',
		type=str,
		default="./data",
		help='The path to the directory containing the data to use (default: ./data)'
	)
	parser.add_argument(
		'-s',
		'--source-file',
		type=str,
		default=None,
		help='The path to the source file to use to compute sentence statistics (default: None)'
	)

	parser.add_argument(
		'--model-name',
		type=str,
		default=None,
		help='The name of the model to be used for embeddings in some cases (default: None)'
	)

	parser.add_argument(
		'--action',
		type=str,
		default='build_wikipedia_sentences',
		choices=[
			'build_wikipedia_sentences', 'build_bookcorpusopen_sentences',
			'compute_sentence_statistics',
			'encapsulate_sentence_dataset',
			'combine_sentence_datasets',
			'build_splits',
			'build_piece_dataset', 'build_sentence_piece_dataset',
			'train-sae', 'test-sae'
		],
		help='The action to perform (default: build_sentence_dataset)'
	)
	parser.add_argument(
		'--max-length',
		type=int,
		default=512,
		help='The maximum length of the input sequence (default: 512)'
	)

	parser.add_argument(
		'--embedding-dim',
		type=int,
		default=768,
		help='The embedding dimension of each token fed into the transformer (default: 768)'
	)
	parser.add_argument(
		'--transformer-heads',
		type=int,
		default=12,
		help='The number of heads to use in each of the encoder and decoder (default: 12, to go with default --embedding-dim=768)'
	)
	parser.add_argument(
		'--embedding-width',
		type=int,
		default=1,
		help='The number of tokens to try to embed the sentence into (default: 1)'
	)
	parser.add_argument(
		'--embedding-multiplier',
		type=int,
		default=1,
		help='The number of times to use the encoder output sentence embedding in the input to the decoder. If -1, the decoder input is filled with the encoder output (default: 1)'
	)
	parser.add_argument(
		'--transformer-depth',
		type=int,
		default=1,
		help='The number of layers to use in each of the encoder and decoder (default: 1)'
	)
	parser.add_argument(
		'--lr',
		type=float,
		default=1e-4,
		help='The learning rate to use in training (default: 1e-4)'
	)
	parser.add_argument(
		'--gradient-accumulation-steps',
		type=int,
		default=1,
		help='The number of steps to accumulate gradients over before taking an optimizer step (default: 1)'
	)
	parser.add_argument(
		'--epochs',
		type=int,
		default=1,
		help='The number of epochs to train for (default: 1)'
	)
	parser.add_argument(
		'--batch-size',
		type=int,
		default=16,
		help='The batch size to use for training and evaluation (default: 16)'
	)


	parser.add_argument(
		'--seed',
		type=int,
		default=42,
		help='The seed for torch, numpy, and python randomness (default: 1234)'
	)
	
	parser.add_argument(
		'--checkpoint-frequency',
		type=int,
		default=1_000_000,
		help='The frequency at which to save checkpoints (default: 100000)'
	)
	parser.add_argument(
		'--checkpoint',
		type=str,
		default=None,
		help='The name of the checkpoint to load (default: None)'
	)

	return parser