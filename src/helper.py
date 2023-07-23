import pandas as pd

def load_data(fn, label):
    data = []
    with open(fn) as fp:
        q = ""
        for line in fp:
            line = line.rstrip()
            if line != "":
                if q:
                    q += " " + line
                else:
                    q = line
            else:
                data.append((q, label))
                q = ""
    return data

def load_quora_data(fn):
    df = pd.read_csv(fn)
    q1 = df["question1"].tolist()
    q2 = df["question2"].tolist()
    q1_ = [(q, 'other') for q in q1]
    q2_ = [(q, 'other') for q in q2]
    return q1_+q2_

def create_ft_data(raw_data, out_fn):
    data = []
    for item in raw_data:
        q, label = item
        txt = "__label__" + label + " " + q
        data.append(txt)
    with open(out_fn, 'w') as fp:
        for txt in data:
            fp.write(txt + "\n")
