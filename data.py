import os
import sys
import datasets
import nltk
from tqdm import tqdm
nltk.download('punkt')
import traceback

import numpy as np
from scipy import stats
from pathlib import Path

def form_source_dataset(meta_config):
    # use hugginface.datasets to load the wikipedia en and bookcorpusopen datasets, then concatenate them
	bookcorpusopen = datasets.load_dataset("bookcorpusopen", split="train", cache_dir=meta_config.input_path)
	wikipedia = datasets.load_dataset("wikipedia", "20220301.en", split="train", cache_dir=meta_config.input_path)

	# concatenate the datasets
	combined_dataset = datasets.concatenate_datasets([ bookcorpusopen, wikipedia ])
	combined_dataset = combined_dataset.with_format("torch")

	combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
	return combined_dataset

def build_piece_dataset(meta_config, source_dataset, tokenizer):
	source_dataset.set_transform(lambda row: tokenizer(row['text']))
	
	with open(os.path.join(meta_config.input_path, "wiki_and_bco-piece.txt"), 'w', encoding="utf-8", buffering=16*1024*1024) as f:
		for i in tqdm(range(len(source_dataset))):
			document_len = len(source_dataset[i]['input_ids'])
			for j in range(0, 1 + document_len // 512):
				start = j * 512
				end = min((j+1) * 512, document_len)
				if start == end: # in case the length is exactly a multiple of 512
					break
				f.write(tokenizer.decode(source_dataset[i]['input_ids'][start:end], skip_special_tokens=True) + "\n")

		f.flush()

def build_sentence_dataset(meta_config, source_dataset, source_dataset_name):
	with open(os.path.join(meta_config.input_path, f"{source_dataset_name}-sentences-{meta_config.job_id}.txt"), 'w', encoding="utf-8", buffering=4*1024*1024) as f:
		for i in tqdm(range(len(source_dataset))):
			sentences = nltk.sent_tokenize(source_dataset[i]['text'])
			for sentence in sentences:
				if sentence.strip() == "" or len(sentence) > 512:
					continue
				sentence_nl_escaped = sentence.replace("\n", "\\n")
				f.write(sentence_nl_escaped + "\n")

		f.flush()

	print("Raw sentence dataset built into the file '" + f"{source_dataset_name}-sentences-{meta_config.job_id}.txt")

def encapsulate_sentence_dataset(meta_config, dataset_file_to_load):
	dataset = load_sentence_dataset(meta_config, dataset_file_to_load)
	print(f"Loaded {dataset_file_to_load}")
	input_filename = Path(dataset_file_to_load).stem
	print(f"Processing dataset {input_filename}")
	dst_path = os.path.join(meta_config.input_path, f"{input_filename}-encapsulated-{meta_config.job_id}")
	dataset.save_to_disk(dst_path)
	print(f"Encapsulated dataset saved to {dst_path}")


def load_sentence_dataset(meta_config, dataset_file_to_load):
	dataset = datasets.load_dataset('text', data_files=[ os.path.join(meta_config.input_path, dataset_file_to_load) ], split='train', cache_dir=meta_config.input_path)
	return dataset

def update_histogram(array: np.array, value: int):
	if value >= len(array):
		array = np.concatenate((array, np.zeros(value - len(array) + 1)))
	array[value] += 1
	return array

def compute_sentence_statistics(meta_config, source_dataset, tokenizer):
	char_lengths = []
	word_lengths = []
	token_lengths = []

	char_histogram = np.array([])
	word_histogram = np.array([])
	token_histogram = np.array([])

	for i in tqdm(range(len(source_dataset))):
		row = source_dataset[i]
		row_text = row['text'].replace("\\n", "\n")
		words = nltk.word_tokenize(row_text)
		tokens = tokenizer([row_text], truncation=False)['input_ids'][0]

		char_lengths.append(len(row_text))
		char_histogram = update_histogram(char_histogram, len(row_text))
		word_lengths.append(len(words))
		word_histogram = update_histogram(word_histogram, len(words))
		token_lengths.append(len(tokens))
		token_histogram = update_histogram(token_histogram, len(tokens))
	
	char_lengths = np.array(char_lengths)
	word_lengths = np.array(word_lengths)
	token_lengths = np.array(token_lengths)

	print("char stats:", stats.describe(char_lengths))
	print("median char length", np.median(char_lengths))
	print("1/4 quantile char length", np.quantile(char_lengths, 0.25))
	print("3/4 quantile char length", np.quantile(char_lengths, 0.75))
	print("iqr char length", np.quantile(char_lengths, 0.75) - np.quantile(char_lengths, 0.25))
	print("95 precentile char length", np.quantile(char_lengths, 0.95))
	print("99 precentile char length", np.quantile(char_lengths, 0.97))

	print("word stats:", stats.describe(word_lengths))
	print("median word length", np.median(word_lengths))
	print("1/4 quantile word length", np.quantile(word_lengths, 0.25))
	print("3/4 quantile word length", np.quantile(word_lengths, 0.75))
	print("iqr word length", np.quantile(word_lengths, 0.75) - np.quantile(word_lengths, 0.25))
	print("95 precentile word length", np.quantile(word_lengths, 0.95))
	print("99 precentile word length", np.quantile(word_lengths, 0.97))

	print("token stats:", stats.describe(token_lengths))
	print("median token length", np.median(token_lengths))
	print("1/4 quantile token length", np.quantile(token_lengths, 0.25))
	print("3/4 quantile token length", np.quantile(token_lengths, 0.75))
	print("iqr token length", np.quantile(token_lengths, 0.75) - np.quantile(token_lengths, 0.25))
	print("95 precentile token length", np.quantile(token_lengths, 0.95))
	print("99 precentile token length", np.quantile(token_lengths, 0.97))

	with np.printoptions(threshold=sys.maxsize):
		print("char histogram:")
		print(char_histogram[0:min(512, char_histogram.size)])
		print("char histogram overflow:", np.sum(word_histogram[512:]))
		print("word histogram:")
		print(word_histogram[0:min(128, word_histogram.size)])
		print("word histogram overflow:", np.sum(word_histogram[128:]))
		print("token histogram:")
		print(token_histogram[0:min(128, token_histogram.size)])
		print("token histogram overflow:", np.sum(token_histogram[128:]))

def build_sentence_piece_dataset(meta_config, source_dataset, tokenizer):
	MAX_SIZE = 2048
	with open(os.path.join(meta_config.input_path, f"wiki_and_bco-sentence_piece-{MAX_SIZE}.txt"), 'w', encoding="utf-8", buffering=16*1024*1024) as f:
		for i in tqdm(range(len(source_dataset))):
			sentences = nltk.sent_tokenize(source_dataset[i]['text'])
			running_input_ids = []
			for sentence in sentences:
				sentence_input_ids = tokenizer(sentence)['input_ids']
				sentence_len = len(sentence_input_ids)
				if len(sentence_input_ids) > MAX_SIZE:
					for j in range(0, 1 + sentence_len // MAX_SIZE):
						start = j * MAX_SIZE
						end = min((j+1) * MAX_SIZE, sentence_len)
						if start == end: # in case the length is exactly a multiple of MAX_SIZE
							break
						f.write(tokenizer.decode(sentence_input_ids[start:end], skip_special_tokens=True) + "\n")
				else:
					if len(running_input_ids) + len(sentence_input_ids) < MAX_SIZE:
						running_input_ids += sentence_input_ids
					else:
						f.write(tokenizer.decode(running_input_ids, skip_special_tokens=True) + "\n")
						running_input_ids = sentence_input_ids

			if len(running_input_ids) > 0:
				f.write(tokenizer.decode(running_input_ids, skip_special_tokens=True) + "\n")

def load_dataset(meta_config, tokenizer):
	if meta_config.action == 'train-sae' or meta_config.action == 'test-sae':
		dataset = datasets.load_from_disk(os.path.join(meta_config.input_path, meta_config.source_file))
		if meta_config.action == 'train-sae':
			dataset = dataset['train']
		else:
			dataset = dataset['test']
		dataset.set_transform(lambda row: tokenizer(row['text'], truncation=True, max_length=meta_config.max_length))
		dataset = dataset.shuffle(seed=meta_config.seed)
	else:
		dataset_file_to_load = f"wiki_and_bco-sentence_piece-{meta_config.max_length}.txt"
		dataset = datasets.load_dataset('text', data_files=[ os.path.join(meta_config.input_path, dataset_file_to_load) ], split='train', cache_dir=meta_config.input_path)
		dataset.set_transform(lambda row: sentence_piece_transform(meta_config, tokenizer, row))
		dataset = dataset.shuffle(seed=meta_config.seed)

	return dataset

def combine_sentence_datasets(meta_config):
	source_files = meta_config.source_file.split(',')
	print(f"Loading wikipedia from {source_files[0]}")
	wikipedia = datasets.load_from_disk(os.path.join(meta_config.input_path, source_files[0]))
	print(f"Loading bookcorpusopen from {source_files[1]}")
	bookcorpusopen = datasets.load_from_disk(os.path.join(meta_config.input_path, source_files[1]))

	combined_dataset = datasets.concatenate_datasets([ bookcorpusopen, wikipedia ])
	combined_dataset = combined_dataset.with_format("torch")

	combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
	combined_dataset.save_to_disk(os.path.join(meta_config.input_path, f"wikipedia-and-bco-sentences-{meta_config.job_id}"))

def build_splits(meta_config):
	print(f"Loading dataset from {meta_config.source_file}")
	dataset = datasets.load_from_disk(os.path.join(meta_config.input_path, meta_config.source_file))
	print(f"Splitting dataset")
	split_dataset = dataset.train_test_split(test_size=int(1e6))

	source_file_stem = Path(meta_config.source_file).stem
	print(f"Saving dataset to {source_file_stem}-split")
	split_dataset.save_to_disk(os.path.join(meta_config.input_path, source_file_stem + "-split"))

def sentence_piece_transform(meta_config, tokenizer, row):
	assert len(row['text']) == 1

	if 'text' not in row.keys() or len(row['text'][0]) == 0:
		row['text'] = [ "Mary had a little lamb." ] # dummy text is better than returning None or whatever and invoking an undefined behaviour

	sentences = nltk.sent_tokenize(row['text'][0])
	try:
		ret = tokenizer(sentences, truncation=True, max_length=meta_config.max_length)
	except IndexError as e:
		print(e)
		traceback.print_exc()
		print("Deliquent sentence list: ", sentences)
		print("Deliquent entry passed from Dataset: ", row['text'])
	
	ret['input_ids'] = [ ret['input_ids'] ]
	ret['attention_mask'] = [ ret['attention_mask'] ]

	return ret