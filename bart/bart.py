from transformers import AutoTokenizer, BartForConditionalGeneration
import os

def get_model_tokenizer():
    """ Setup and return the model and its tokenizer"""
    model_dir = "bart/bart_model"
    checkpoint = "facebook/bart-large-cnn"

    if not os.path.exists(model_dir):
        print(f"Model {model_dir} does not exist. It will be downloaded from Huggingface")
        os.makedirs(model_dir)

        model = BartForConditionalGeneration.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        print(f"Model {model_dir} stored locally. This local version will be uploaded")
        model = BartForConditionalGeneration.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return tokenizer, model

def bart_pipeline():
    from transformers import pipeline

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    article = """
        PG&E stated it scheduled the blackouts in response to forecasts for high winds 
        amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were 
        scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.
    """
    print(summarizer(article, max_length=130, min_length=30, do_sample=False))


def bart_stages():
    """ Setup model and its tokenizer"""
    tokenizer, model = get_model_tokenizer()
    article = """
        PG&E stated it scheduled the blackouts in response to forecasts for high winds 
        amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were 
        scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.
    """

    """ Stage-1: Pre-processing """
    inputs = tokenizer(article, max_length=1024, return_tensors="pt")

    """ Stage-2: Model """
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)

    """ Stage-3: Postprocessing """
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(summary)

def main():
    # bart_pipeline()
    bart_stages()


if __name__ == "__main__":
    main()