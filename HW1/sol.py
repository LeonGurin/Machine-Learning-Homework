import readTrainData




def main():
    texAll, lbAll, voc, cat = readTrainData("train.txt")
    Pw, P = learn_NB_text()
    



if __name__ == "__main__":
    main()

