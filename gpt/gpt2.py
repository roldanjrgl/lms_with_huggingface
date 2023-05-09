from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import pipeline, set_seed

def gpt2_pipelines():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    print(output)

def gpt2_stages():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    """ Stage-1: Pre-processing """
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')

    generated_sequence = model.generate(**encoded_input)
    # output = model(**encoded_input)
    print(generated_sequence)
    out_b = generated_sequence.shape[0]
    in_b = encoded_input["input_ids"].shape[0]
    framework = "pt"
    if framework == "pt":
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])

    text = tokenizer.decode(
        generated_sequence,
        skip_special_tokens=True,
        # clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
    print(text)

def main():
    gpt2_pipelines()
    # gpt2_stages()


if __name__ == "__main__":
    main()
