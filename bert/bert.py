from transformers import BertTokenizer, BertForSequenceClassification


def bert_stages():
    """ Setup model and its tokenizer"""
    checkpoint = "yiyanghkust/finbert-tone"
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    model_name = 'bert_pytorch'
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    text = "Spring is so beautiful"

    """ Stage-1: Pre-processing """
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