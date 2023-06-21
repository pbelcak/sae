# SAE

```
usage: __main__.py [-h] [-j JOB_ID] [-o OUTPUT_PATH] [-i INPUT_PATH] [-s SOURCE_FILE] [--model-name MODEL_NAME]
                   [--action {build_wikipedia_sentences,build_bookcorpusopen_sentences,compute_sentence_statistics,encapsulate_sentence_dataset,combine_sentence_datasets,build_splits,build_piece_dataset,build_sentence_piece_dataset,train-sae,test-sae}]
                   [--max-length MAX_LENGTH] [--embedding-dim EMBEDDING_DIM] [--transformer-heads TRANSFORMER_HEADS]
                   [--embedding-width EMBEDDING_WIDTH] [--embedding-multiplier EMBEDDING_MULTIPLIER] [--transformer-depth TRANSFORMER_DEPTH]
                   [--lr LR] [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                   [--seed SEED] [--checkpoint-frequency CHECKPOINT_FREQUENCY] [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -j JOB_ID, --job-id JOB_ID
                        The job id (and the name of the wandb group)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        The directory which will contain saved model checkpoints
  -i INPUT_PATH, --input-path INPUT_PATH
                        The path to the directory containing the data to use (default: ./data)
  -s SOURCE_FILE, --source-file SOURCE_FILE
                        The path to the source file to use to compute sentence statistics (default: None)
  --model-name MODEL_NAME
                        The name of the model to be used for embeddings in some cases (default: None)
  --action {build_wikipedia_sentences,build_bookcorpusopen_sentences,compute_sentence_statistics,encapsulate_sentence_dataset,combine_sentence_datasets,build_splits,build_piece_dataset,build_sentence_piece_dataset,train-sae,test-sae}
                        The action to perform (default: build_sentence_dataset)
  --max-length MAX_LENGTH
                        The maximum length of the input sequence (default: 512)
  --embedding-dim EMBEDDING_DIM
                        The embedding dimension of each token fed into the transformer (default: 768)
  --transformer-heads TRANSFORMER_HEADS
                        The number of heads to use in each of the encoder and decoder (default: 12, to go with default --embedding-dim=768)
  --embedding-width EMBEDDING_WIDTH
                        The number of tokens to try to embed the sentence into (default: 1)
  --embedding-multiplier EMBEDDING_MULTIPLIER
                        The number of times to use the encoder output sentence embedding in the input to the decoder. If -1, the decoder input is
                        filled with the encoder output (default: 1)
  --transformer-depth TRANSFORMER_DEPTH
                        The number of layers to use in each of the encoder and decoder (default: 1)
  --lr LR               The learning rate to use in training (default: 1e-4)
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        The number of steps to accumulate gradients over before taking an optimizer step (default: 1)
  --epochs EPOCHS       The number of epochs to train for (default: 1)
  --batch-size BATCH_SIZE
                        The batch size to use for training and evaluation (default: 16)
  --seed SEED           The seed for torch, numpy, and python randomness (default: 1234)
  --checkpoint-frequency CHECKPOINT_FREQUENCY
                        The frequency at which to save checkpoints (default: 100000)
  --checkpoint CHECKPOINT
                        The name of the checkpoint to load (default: None)
```