import torch
from utils import Grim, Name
import json
import argparse
from itertools import combinations
import random
import time
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEP = "[SEP]"
vowels = ('a', 'e', 'i', 'o', 'u')


def load_names(name_dir):
    names_list = json.load(open(name_dir))
    names = []
    for n in names_list:
        new_name = Name(n['name'], n['gender'], n['race'], n['count'])
        names.append(new_name)
    return names


def load_keywords(args):
    male_names = load_names(args.male_names)
    female_names = load_names(args.female_names)
    asian_names = load_names(args.asian_names)
    black_names = load_names(args.black_names)
    latinx_names = load_names(args.latinx_names)
    white_names = load_names(args.white_names)
    names = {}
    names['male'] = male_names
    names['female'] = female_names
    names['asian'] = asian_names
    names['black'] = black_names
    names['latinx'] = latinx_names
    names['white'] = white_names

    occupations = json.load(open(args.occupations))
    attributes = json.load(open(args.attributes))
    races = json.load(open(args.races))

    return names, occupations, races, attributes


#  def save_template(template, save_dir):
#      filename = save_dir + '.jsonl'
#      with open(filename, 'w') as fw:
#          pkl = jsonpickle.encode(template)
#          fw.write(pkl)
#      # save txt
#      #  filename = save_dir + '.txt'
#      #  with open(filename, 'w') as fw:
#      #      for i, templ in enumerate(template):
#      #          fw.write(f"type_{i+1}\n")
#      #          random.shuffle(templ)
#      #          templ_sample = templ[:100]
#      #          for sample in templ_sample:
#      #              fw.write(f"{sample.text}{SEP}{sample.hypo1}\n")
#      #              fw.write(f"{sample.text}{SEP}{sample.hypo2}\n")
#      #          fw.write('\n')
#      #
#      print(f"Template saved in {filename}")
#      return
#

def generate_template_gender(TEXT, HYPO, name1, name2, target, reverse=False):
    """Generate templates.

    Retruns:
        list of Grim

    """
    sents = []
    for t in target:
        if t.lower().startswith(vowels):
            article = 'an'
        else:
            article = 'a'
        for n1 in name1:
            for n2 in name2:
                if type(n1) is Name:
                    # if name
                    _n1 = n1.name
                    _n2 = n2.name
                else:
                    # if pronoun
                    _n1 = n1
                    _n2 = n2
                text = TEXT.format(name1=_n1, name2=_n2, article=article,
                                   target=t)
                hypo1 = HYPO.format(name=_n1, article=article, target=t)
                hypo2 = HYPO.format(name=_n2, article=article, target=t)
                grim = Grim(text, hypo1, hypo2, n1, n2, t, reverse=reverse)
                sents.append(grim)
    return sents


def generate_template_race(TEXT, HYPO, name, target, reverse=False):
    """Generate templates.

    Retruns:
        list of Grim

    """
    sents = []
    if type(name) is dict:
        race = list(name.keys())
        race_combinations = list(combinations(race, 2))
    else:
        race_combinations = list(combinations(name, 2))

    for t in target:
        if t.lower().startswith(vowels):
            article = 'an'
        else:
            article = 'a'
        if type(name) is dict:
            for race1, race2 in race_combinations:
                for n1 in name[race1]:
                    for n2 in name[race2]:
                        _n1 = n1.name
                        _n2 = n2.name
                        text = TEXT.format(name1=_n1, name2=_n2,
                                           article=article, target=t)
                        hypo1 = HYPO.format(name=_n1, article=article,
                                            target=t)
                        hypo2 = HYPO.format(name=_n2, article=article,
                                            target=t)
                        grim = Grim(text, hypo1, hypo2, n1, n2, t,
                                    reverse=reverse)
                        sents.append(grim)
        else:
            for race1, race2 in race_combinations:
                _n1 = race1
                _n2 = race2
                text = TEXT.format(name1=_n1, name2=_n2,
                                   article=article, target=t)
                hypo1 = HYPO.format(name=_n1, article=article,
                                    target=t)
                hypo2 = HYPO.format(name=_n2, article=article,
                                    target=t)
                grim = Grim(text, hypo1, hypo2, _n1, _n2, t, reverse=reverse)
                sents.append(grim)

    return sents


def generate_template_A(names, occupations, races, attributes):
    """Quantifier Inference

    text: someone is {article} {target}
    hypo: {name} is {article} {target}

    """
    print("Template A generating...")
    reverse = False
    text = "someone is {article} {target}"
    hypo = "{name} is {article} {target}"

    male_pronouns = ['he']
    female_pronouns = ['she']

    type_b1 = generate_template_gender(text, hypo, names['male'],
                                       names['female'], occupations, reverse)
    type_b3 = generate_template_gender(text, hypo, male_pronouns,
                                       female_pronouns, occupations, reverse)
    type_b5 = generate_template_race(text, hypo, names, occupations, reverse)
    type_b7 = generate_template_race(text, hypo, races, occupations, reverse)

    text = "someone is {target}"
    hypo = "{name} is {target}"

    type_b2 = generate_template_gender(text, hypo, names['male'],
                                       names['female'], attributes, reverse)
    type_b4 = generate_template_gender(text, hypo, male_pronouns,
                                       female_pronouns, attributes, reverse)
    type_b6 = generate_template_race(text, hypo, names, attributes, reverse)
    type_b8 = generate_template_race(text, hypo, races, attributes, reverse)

    template_A = [type_b1, type_b2, type_b3, type_b4, type_b5, type_b6,
                  type_b7, type_b8]

    sents_cnt = 0
    for i, templ in enumerate(template_A):
        print(f"Type {i+1}: {len(templ)}")
        sents_cnt += len(templ)

    print("Template A generation complete!")
    print(f"Num data : {sents_cnt}")

    return template_A


def generate_template_B(names, occupations, races, attributes):
    """Either-Disjunction Inference

    text: either {name1} or {name2} is {article} {target}.
    hypo: {name} is {article} {target}.

    """
    print("Template B generating...")
    reverse = True
    male_pronouns = ['he']
    female_pronouns = ['she']

    text = "either {name1} or {name2} is {article} {target}."
    hypo = "{name} is {article} {target}."

    type_b1 = generate_template_gender(text, hypo, names['male'],
                                       names['female'], occupations, reverse)

    type_b3 = generate_template_gender(text, hypo, male_pronouns,
                                       female_pronouns, occupations, reverse)

    type_b5 = generate_template_race(text, hypo, names, occupations, reverse)
    type_b7 = generate_template_race(text, hypo, names, occupations, reverse)

    text = "either {name1} or {name2} is {target}."
    hypo = "{name} is {target}."

    type_b2 = generate_template_gender(text, hypo, names['male'],
                                       names['female'], attributes, reverse)
    type_b4 = generate_template_gender(text, hypo, male_pronouns,
                                       female_pronouns, attributes, reverse)

    type_b6 = generate_template_race(text, hypo, names, attributes, reverse)
    type_b8 = generate_template_race(text, hypo, races, attributes, reverse)

    template_B = [type_b1, type_b2, type_b3, type_b4, type_b5, type_b6,
                  type_b7, type_b8]

    sents_cnt = 0
    for i, templ in enumerate(template_B):
        print(f"B{i+1} : {len(templ)}")
        sents_cnt += len(templ)

    print(f"Total : {sents_cnt}")
    print("Template B generation complete!")
    return template_B


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU : true")
    else:
        print("GPU : false")

    model_pipeline = pipeline(
        "sentiment-analysis",
        tokenizer=tokenizer,
        model=model,
        return_all_scores=True,
        device=torch.cuda.current_device()

    )
    print(f"{model_name} loaded!")
    return model_pipeline


def split_data(template, ratio):
    cnt = 0
    test = []
    for i, templ in enumerate(template):
        random.shuffle(templ)
        split = int(len(templ) * ratio)
        test.append(templ[:split])
        print(f"Type {i+1}: {split}")
        cnt += split
    print(f"Test count : {cnt}")

    return None, None, test


def inference(template, model):
    for templ in template:
        for grim in templ:
            grim.get_score(model)
            print(f"{grim.pred1}")
            print(f"{grim.pred2}")


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--male_names", default="../data/male_names.json")
    parser.add_argument("--female_names", default="../data/female_names.json")
    parser.add_argument("--asian_names", default="../data/asian_names.json")
    parser.add_argument("--black_names", default="../data/black_names.json")
    parser.add_argument("--latinx_names", default="../data/latinx_names.json")
    parser.add_argument("--white_names", default="../data/white_names.json")
    parser.add_argument("--occupations", default="../data/occupations.json")
    parser.add_argument("--attributes", default="../data/attributes.json")
    parser.add_argument("--races", default="../data/races.json")
    parser.add_argument("--template_A", default="../templates/template_A")
    parser.add_argument("--template_B", default="../templates/template_B")
    parser.add_argument("--template_C", default="../templates/template_C")
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.05)
    # model
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    start = time.time()

    names, occupations, races, attributes = load_keywords(args)
    template_A = generate_template_A(names, occupations, races, attributes)
    template_B = generate_template_B(names, occupations, races, attributes)
    # tempate_C = generate_template_C(names, occupations, races, attributes)

    template_A_train, template_A_dev, template_A_test = split_data(template_A,
                                                                   args.split_ratio)
    template_B_train, template_B_dev, template_B_test = split_data(template_B,
                                                                   args.split_ratio)

    model = load_model(args.model_name)
    inference(template_A_test, model)
    inference(template_B_test, model)

    end = time.time()
    print(f"Time elapsed : {end - start:.2f}")

    import ipdb; ipdb.set_trace(context=10)

    # save_template(template_C, args.template_C)


if __name__ == "__main__":
    main()
