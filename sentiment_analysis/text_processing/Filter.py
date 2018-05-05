import pandas as pd
import psutil

def filter(data, vocab, out, widget):
    data = data.dropna(axis=0, how='any')
    # limit to words in vocab
    proc = psutil.Process()
    widget.put(("Process " + str(proc.pid) + " pipe is working"))
    print('Process PID: ' + str(proc.pid) + ', Affinity: ' +str(proc.cpu_affinity()))
    fiveperc = len(data) / 20
    i = 0
    for index, row in data.iterrows():
        i += 1
        if (i % fiveperc == 0):
            print("Process " + str(proc.pid) + " is " + str(5 * i / fiveperc) + " percent done")
        tokens = row['text'].split()
        tokens = [w for w in tokens if w in vocab]
        data.at[index, 'text'] = ' '.join(tokens)
    out.put(data)