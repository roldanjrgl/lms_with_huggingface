from transformers import BertTokenizer, BertForSequenceClassification
import os

def get_model_tokenizer():
    """ Setup and return the model and its tokenizer"""
    model_dir = "bert/bert_model"
    checkpoint = "yiyanghkust/finbert-tone"

    if not os.path.exists(model_dir):
        print(f"Model {model_dir} does not exist. It will be downloaded from Huggingface")
        os.makedirs(model_dir)

        model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(checkpoint)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        print(f"Model {model_dir} stored locally. This local version will be uploaded")
        model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(checkpoint)

    return tokenizer, model


def bert_stages():
    tokenizer, model = get_model_tokenizer()

    """ Stage-1: Pre-processing """
    text = "Spring is so beautiful"
    inputs = tokenizer(text, return_tensors = "pt")

    """ Stage-2: Model """
    output = model(**inputs)
    print(output)

    """ Stage-3: Postprocessing """
    # TODO: Finish



def main():
    bert_stages()


if __name__ == "__main__":
    main()