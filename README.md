# Getting Started
![Our proposed model whole architechture](Images/full_image.png) 

## Installation

1. **Prepare the code and environment and install library:**
   - Git clone our repository, create a Python environment, and install the library.

2. **Prepare the pretrained Vicuna weights:**
   - The current version of InVENTO is built on the Vicuna-13B version and Vicuna-7B version. Refer to our [instruction](PrepareVicuna.md) to prepare the Vicuna weights.
   - The final weights should be in a single folder with a structure similar to the following:
     ```plaintext
     vicuna_weights
     ├── config.json
     ├── generation_config.json
     ├── pytorch_model.bin.index.json
     ├── pytorch_model-00001-of-00003.bin
     ...
     ```
   - Set the path to the Vicuna weight in the model config file [here](ad_invento/configs/models/invento.yaml) at Line 14.

3. **Prepare the Evaluation checkpoint:**
   - Download the pretrained checkpoints according to the Vicuna model you prepared.
   - Set the path of the checkpoint after train the model in the evaluation config file in [here](eval_configs/invento_eval.yaml) at Line 11.

## Training

1. **First pretraining stage:**
   - Train the [I-JEPA](https://github.com/facebookresearch/ijepa) model using your dataset and save the checkpoint.
   

2. **Second pretraining stage:**
   - Set the path to the pretrained checkpoint in the [file](ad_invento/models/mymodel.py) at Line 61. Add I-JEPA path at Line 57.
   - download the file from [here](https://drive.google.com/file/d/1StoRiI3S7u3qTbh1GEocCkSGbyDMQY8u/view?usp=drive_link) and then add it in the config [file](train_configs/invento_stage2_finetune.yaml) in Line 21
   - Train the model as mentioned in the [paper](link) using freeze encoder and decoder.
   - You can change the save path in the config file `train_configs/invento_stage2_finetune.yaml`.

4. **Third fine-tune decoder stage:**
   - Train the model as mentioned in the [paper](link).
   - Use Second pretraining stage saved checkpoint for this stage. Change the Line 21 and add the Second pretraining stage checkpoint here 
   - You can change the save path in the config file `train_configs/invento_stage2_finetune.yaml`.

For the second and third stage, use the following command:
   ```bash
   python train.py --cfg-path train_configs/invento_stage2_finetune.yaml --gpu-id 0
   ```

## Evaluation

For the evaluation of the model, check the [notebook](Prediction_Notebook.ipynb).

## Acknowledgement

- **[BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2 ):** Check out this great open-source work if you haven't already!
- **[Vicuna](https://github.com/lm-sys/FastChat):** The fantastic language ability of Vicuna with only 13B parameters is amazing. And it is open-source!
- **[IJEPA](https://github.com/facebookresearch/ijepa):**  Gives a sementic level image representation.

If you're using InVENTO in your research or applications, please cite using this BibTeX:
```plaintext
[Insert BibTeX citation here]
