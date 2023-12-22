# How to Prepare Vicuna Weight

Vicuna is an open-source LLAMA-based LLM that closely matches the performance of ChatGPT. The following steps outline how to prepare Vicuna's weight using the v0 version of Vicuna-13B.

### 1. Download Vicuna's Delta Weight

- First, download Vicuna’s delta weight from [here](https://huggingface.co/lmsys/vicuna-13b-delta-v0).
- If you have git-lfs installed ([git-lfs.com](https://git-lfs.com)), use the following commands:
    ```bash
    git lfs install
    git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0  # more powerful, requires at least 24G GPU memory
    # or
    git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0  # smaller, requires 12G GPU memory
    ```
  Note: This weight represents the difference between the working weight and the original weight of LLAMA-13B. LLAMA's rules prohibit the distribution of its weight directly.

### 2. Obtain Original LLAMA-7B or LLAMA-13B Weights

- Obtain the original LLAMA-7B or LLAMA-13B weights in the HuggingFace format either by following the instructions provided by HuggingFace [here](https://huggingface.co/docs/datasets/loading_datasets.html) or from the Internet.

### 3. Create the Working Weight

- Install the library compatible with v0 Vicuna by running:
    ```bash
    pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
    ```
- Use the following command to create the final working weight:
    ```bash
    python -m fastchat.model.apply_delta --base /path/to/llama-13bOR7b-hf/ --target /path/to/save/working/vicuna/weight/ --delta /path/to/vicuna-13bOR7b-delta-v0/
    ```

Now you have prepared the working weight for Vicuna!
