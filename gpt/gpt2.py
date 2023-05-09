import os

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import pipeline, set_seed
import os

def get_model_tokenizer():
    """ Setup and return the model and its tokenizer"""
    model_dir = "gpt/gpt2_model"
    checkpoint = "gpt2"

    if not os.path.exists(model_dir):
        print(f"Model {model_dir} does not exist. It will be downloaded from Huggingface")
        os.makedirs(model_dir)

        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        model = GPT2LMHeadModel.from_pretrained(checkpoint)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        print(f"Model {model_dir} stored locally. This local version will be uploaded")
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    return tokenizer, model


def gpt2_pipelines():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    print(output)

def gpt2_stages():
    tokenizer, model = get_model_tokenizer()

    """ Stage-1: Pre-processing """
    text = "Today is a beautiful day, I will"
    encoded_input = tokenizer(text, return_tensors='pt')
    # input_ids = encoded_input["input_ids"]
    # attention_mask = encoded_input["attention_mask"]

    """ Stage-2: Model """
    generated_sequence = model.generate(**encoded_input)
    generated_sequence = generated_sequence[0]
    print(generated_sequence)
    out_b = generated_sequence.shape[0]
    in_b = encoded_input["input_ids"].shape[0]
    framework = "pt"
    if framework == "pt":
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    generated_sequence = generated_sequence.numpy().tolist()

    """ Stage-3: Postprocessing """
    text = tokenizer.decode(
        generated_sequence[0],
        skip_special_tokens=True,
        # clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
    print(text)

def main():
    # gpt2_pipelines()
    gpt2_stages()


if __name__ == "__main__":
    main()
