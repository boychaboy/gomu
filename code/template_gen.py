import json
from utils import Gomu, Name


def load_keywords(data_dir):
    names_list = json.load(open(data_dir + 'names.json'))
    names = []
    for n in names_list:
        new_name = Name(n[0], n[1], n[2])
        names.append(new_name)
    occupations = json.load(open(data_dir + 'occupations.json'))
    race = json.load(open(data_dir + 'race.json'))
    attributes = json.load(open(data_dir + 'attribute.json'))

    return names, occupations, race, attributes


 __name__ == "__main__":
    data_dir = '../data/'
    names, occupations, race, attributes = load_keywords(data_dir)
    template_A, tempalte_B, template_C = load_templates(data_dir)


