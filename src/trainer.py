import helper
import random
import fasttext


def load_train_data(filenames):
    # Load files
    raw_data = []
    for label, fn in filenames.items():
        data = helper.load_data(fn, label)
        raw_data += data

    return raw_data


def main():
    math_fn = 'data/math.txt'
    phys_fn = 'data/phys.txt'
    chem_fn = 'data/chem.txt'
    filenames = {'math': math_fn, 'phys': phys_fn, 'chem': chem_fn}
    raw_data = load_train_data(filenames)
    print(raw_data[0])
    print(raw_data[1])
    print(raw_data[-1])
    print("Len of stem data: {}".format(len(raw_data)))
    random.shuffle(raw_data)
    other_data = helper.load_quora_data('data/others.txt')
    print(other_data[0])
    print(other_data[-1])
    print(len(other_data))
    random.shuffle(other_data)
    raw_data += other_data[:len(raw_data)]
    random.shuffle(raw_data)
    print(len(raw_data))
    # Create FT dataset
    train_fn = 'data/stem.train'
    helper.create_ft_data(raw_data, train_fn)
    model = fasttext.train_supervised(input=train_fn, lr=1.0, epoch=25, wordNgrams=2)
    model.save_model("models/model_stem.bin")
    print(model.predict("Which baking dish is best to bake a banana bread ?"))
    print(model.predict("The circumference of a circle is 30. What is its area? 15pi 225pi 400pi 900pi 3000pi"))
    test = """An intensity of 60 decibels is ___ times as intense as an intensity of 30 decibels. A. 2 B. 30 C. 60 D. 90 E. 1000"""
    print(model.predict(test))
    test = "In a flame test, the presence of copper in a solution is evident by what color flame? Is the flame w) red x) orange y) indigo z) blue-green"
    print(model.predict(test))
    test = "Compute the largest root of x4 − x3 − 5x2 + 2x + 6."
    print(model.predict(test))


main()