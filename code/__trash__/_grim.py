from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import jsonpickle


def load_template(template_dir):
    with open(template_dir, 'r') as f:
        pkl = f.readline()
        template = jsonpickle.decode(pkl)
    print(f"{template_dir} loaded!")
    return template


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model_pipeline = pipeline(
        "sentiment-analysis",
        tokenizer=tokenizer,
        model=model,
        return_all_scores=True
    )
    print(f"{model_name} loaded!")
    return model_pipeline


def evaluate(template, model):
    scores = []
    for i, templ in enumerate(template):
        score = 0

    return None


def save_result(result, output_dir):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    template = load_template(args.template_dir)
    model = load_model(args.model_name)
    inference(template, model)
    save_result(result, args.output_dir)
    import ipdb; ipdb.set_trace(context=1)



if __name__ == "__main__":
    main()
