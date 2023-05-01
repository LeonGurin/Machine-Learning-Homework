from readTrainData import *
import numpy as np

def learn_NB_text():
    P = []
    Pw = []
    texAll, lbAll, voc, cat = readTrainData("r8-train-stemmed.txt")

    number_of_documents = len(texAll)
    number_of_categories = len(cat)
    number_of_words = len(voc)

    # get number of statements in each category
    category_occurrence_dict = {
        "trade": 0,
        "crude": 0,
        "earn": 0,
        "interest": 0,
        "ship": 0,
        "grain": 0,
        "money-fx": 0,
        "acq": 0
    }

    for i in range(0, number_of_documents):
        category_occurrence_dict[lbAll[i]] += 1

    # calculate P(trade), P(crude), P(earn), P(interest), P(ship), P(grain), P(money-fx), P(acq)
    cat_list = list(category_occurrence_dict.keys())
    for i in range(0, number_of_categories):
        P.append(category_occurrence_dict[cat_list[i]] / number_of_documents)

    total_words_in_category = {
        "trade": 0,
        "crude": 0,
        "earn": 0,
        "interest": 0,
        "ship": 0,
        "grain": 0,
        "money-fx": 0,
        "acq": 0
    }

    word_occurrence_in_category = {
        "trade": [0] * number_of_words,
        "crude": [0] * number_of_words,
        "earn": [0] * number_of_words,
        "interest": [0] * number_of_words,
        "ship": [0] * number_of_words,
        "grain": [0] * number_of_words,
        "money-fx": [0] * number_of_words,
        "acq": [0] * number_of_words
    }

    # make the first row of Pw the words themselves
    voc_list = list(voc)
    Pw.append(voc_list)

    # calculate total_words_in_category and word_occurrence_in_category
    for i in range(0, number_of_documents):
        for j in range(0, number_of_words):
            if voc_list[j] in texAll[i]:
                total_words_in_category[lbAll[i]] += 1
                word_occurrence_in_category[lbAll[i]][j] += 1

    # calculate P(w|trade), P(w|crude), P(w|earn), P(w|interest), P(w|ship), P(w|grain), P(w|money-fx), P(w|acq)
    for i in range(0, number_of_categories):
        row = []
        for j in range(0, number_of_words):
            row.append((word_occurrence_in_category[cat_list[i]][j] + 1) / (total_words_in_category[cat_list[i]] + number_of_words))
        Pw.append(row)

    return Pw, P

def ClassifyNB_text(Pw, P):
    texAll, lbAll, voc, cat = readTrainData("r8-test-stemmed.txt")
    number_of_documents = len(texAll)
    number_of_categories = len(cat)

    cat_list = ["trade", "crude", "earn", "interest", "ship", "grain", "money-fx", "acq"]
    correct = 0

    # classify each document and calculate the accuracy
    for i in range(0, number_of_documents):
        max_prob = None
        max_cat = ""
        for j in range(0, number_of_categories):
            prob = np.log(P[j])
            for k in range(0, len(texAll[i])):
                # if the word is not in the vocabulary, ignore it
                try:
                    index =  Pw[0].index(texAll[i][k])
                    prob += np.log(Pw[j + 1][index])
                except:
                    pass
            if max_prob == None or prob > max_prob:
                max_prob = prob
                max_cat = cat_list[j]
        if max_cat == lbAll[i]:
            correct += 1

    return correct / number_of_documents

if __name__ == "__main__":
    Pw, P = learn_NB_text()
    suc = ClassifyNB_text(Pw, P)
    print("success rate: " + str(suc))
