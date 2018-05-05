from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import sys
from TopicModelingMultiProcessing import GetTopics

def main():
	print("Reading Data")
	data = pd.read_csv(sys.argv[1])
	print("Aggregating By Business ID")

	index = 0
	for index, row in data.iterrows():
		if not(row['business_id'] in structure.keys()):
			structure[row['business_id']] = [row['text']]
		else:
			structure[row['business_id']].append(row['text'])

	data = pd.DataFrame(list(structure.items()), columns=['business_id', 'reviews'])

	print(data)

	print("Spawning Processes")
    n_jobs = 12
    inputs = np.array_split(data, n_jobs)
    tasks = []
    j = 0
    out = mp.Queue()
    widgets = mp.Queue()
    processes = []
    results = []

    for i in inputs:
        processes.append(mp.Process(target=GetTopics, args=(i,out,widgets)))

    for p in processes:
        p.start()
    bars = [widgets.get() for p in processes]
    for f in bars:
        print(f)
        
    results = [out.get() for p in processes]
    print("Joining Processes")
    for p in processes:
        p.join()
    frames = []
    for result in results:
        frames.append(result)
        
    data = pd.concat(frames) 

    data.to_csv("topictagged_businesses.csv")

if __name__ == '__main__':
	main()