class Grim(object):
    def __init__(self, text, hypo1, hypo2, name1, name2, target, reverse=False,
                 score1=None, score2=None):
        """
        NLI data pair
        """
        self.text = text
        self.hypo1 = hypo1
        self.hypo2 = hypo2
        self.name1 = name1
        self.name2 = name2
        self.target = target
        self.score1 = score1
        self.score2 = score2
        self.pred1 = None
        self.pred2 = None
        self.reverse = reverse

    def __get_pred(self, score):
        max_score = 0
        for key in score.keys():
            if max_score < score[key]:
                max_score = score[key]
                pred = key
        return pred

    def get_score(self, model):
        output1 = model(f"{self.text}[SEP]{self.hypo1}")
        self.score1 = dict()
        for out in output1[0]:
            self.score1[out['label']] = out['score']
        output2 = model(f"{self.text}[SEP]{self.hypo2}")
        self.score2 = dict()
        for out in output2[0]:
            self.score2[out['label']] = out['score']
        self.pred1 = self.__get_pred(self.score1)
        self.pred2 = self.__get_pred(self.score2)
        return


class Name(object):
    def __init__(self, name, gender, race, count=-1):
        """
        Name data
        """
        self.name = name
        self.gender = gender
        self.race = race
        self.count = count
