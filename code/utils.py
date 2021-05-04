class Gomu(object):
    def __init__(self, text, hypo1, hypo2, name1, name2, score1, score2):
        """
        NLI data pair
        """
        self.text = text
        self.hypo1 = hypo1
        self.hypo2 = hypo2
        self.name1 = name1
        self.name2 = name2
        self.score1 = score1
        self.score2 = score2
        self.pred1 = self.__get_pred(score1)
        self.pred2 = self.__get_pred(score2)

    def __get_pred(self, score):
        max_score = 0
        for key in score.keys():
            if max_score < score[key]:
                max_score = score[key]
                pred = key
        return pred


class Name(object):
    def __init__(self, name, gender, race):
        """
        Name data
        """
        self.name = name
        self.gender = gender
        self.race = race
