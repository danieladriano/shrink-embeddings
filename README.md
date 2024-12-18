# Shrink Embeddings

Code examples to use matryoshka representation learning, binary quantization, and scalar quantization.

## How to run

To create the environment, use [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

```
uv sync
```

To run:

```
uv run <PYTHON_SCRIPT>
```

### Notes

You can get the index embeddings here: [Hugginface - quantized retrieval](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval/tree/main)

If you are getting some trouble to use `faiss` in mac-os, set this env vars:

```
export OMP_NUM_THREADS=1
export PYTORCH_MPS_DISABLE=1
export LLAMA_NO_METAL=1
export KMP_DUPLICATE_LIB_OK=TRUE
```
