import readTrainData as rdt

def learn_NB_text():
    return

def ClassifyNB_text(Pw, P):
    return

def main():
    texAll, lbAll, voc, cat = rdt.readTrainData("r8-train-stemmed.txt")

    Pw, P = learn_NB_text()
    suc = ClassifyNB_text(Pw, P)
    print(suc)

if __name__ == "__main__":
    main()

