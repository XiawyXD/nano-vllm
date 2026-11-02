# Third-party notices

## Nano-vLLM

This repository extends [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), based on upstream commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`. The original MIT license and copyright notice are preserved in [LICENSE](LICENSE).

## vLLM AWQ / Marlin references

The AWQ packing/reference adapter and Marlin metadata transformations reference or adapt the vLLM project's v0.17.1 AWQ/Marlin implementation, distributed under Apache License 2.0. See [vLLM source](https://github.com/vllm-project/vllm/tree/v0.17.1/vllm/model_executor/layers/quantization) and the [included license](licenses/vllm-Apache-2.0.txt).

The source adapters identify the reference version. Native Marlin execution loads a separately obtained, hash-checked extension; this repository does not redistribute the compiled wheel or native shared library.

## External dependencies and models

PyTorch, Triton, FlashAttention, Transformers, FlashInfer, AutoRound and model checkpoints are obtained separately and retain their respective licenses. The repository does not include model weights. Keep the relevant upstream notices when redistributing separately obtained dependencies or model files.
