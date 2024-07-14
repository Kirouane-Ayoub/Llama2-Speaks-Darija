# Build and Train Llama2🦙 Model from Scratch :  

Welcome to the **[Llama2 Speaks Darija](https://medium.com/@ayoubkirouane3/llama-2-speaks-darija-from-scratch-to-darija-mastery-46157049ef9a)** project! 

This initiative aims to **build a version** of **[Llama2](https://arxiv.org/pdf/2307.09288)** that **understands** and **generates** **Algerian Darija**. By **constructing Llama2** from the **ground up**, we'll delve into both theoretical aspects and practical coding steps, making this project an exciting journey for anyone interested in natural language processing and AI .

The project Details can be found in the This [blog post](https://medium.com/@ayoubkirouane3/llama-2-speaks-darija-from-scratch-to-darija-mastery-46157049ef9a) .

## Dataset Challenges:

[Our dataset](https://huggingface.co/datasets/ayoubkirouane/Algerian-Darija/viewer/default/v1) for training **Llama2** in **Darija** is not perfect; it lacks in both quantity and quality, and isn't fully cleaned. however, it's a start, and together we can improve it to build a better AI that understands our language. let's take on this challenge and make a difference .

## Running the train Script : 

To run the train script, you can use the command line to specify the training parameters.

#### Command-line Arguments

- `--n_epochs`: Number of epochs to train the model (default: 10)
- `--train_data_path`: Path to the training data (default: set in settings.py)
- `--eval_data_path`: Path to the evaluation data (default: set in settings.py)
- `--tokenizer_name`: Name of the tokenizer (default: set in settings.py)

#### Example : 

```bash
python train.py --n_epochs 20 --train_data_path train_eval_data/train_data.txt --eval_data_path train_eval_data/eval_data.txt --tokenizer_name "hf-internal-testing/llama-tokenizer"
```


## Running the Inference Script : 

This will generate and print multiple sequences of text based on the provided prompt.

#### Command-line Arguments

- `--prompt`: Input prompt text in Algerian Darija .
- `--model_path`: Path to the saved model file .
- `--max_length`: Maximum length of generated text (default: 30) .
- `--num_return_sequences`: Number of sequences to generate (default: 5) .

#### Example : 
```bash
python inference.py --prompt "وحد نهار" --model_path "model_final.pt" --max_length 30 --num_return_sequences 5
```
