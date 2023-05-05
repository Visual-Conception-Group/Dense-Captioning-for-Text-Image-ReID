import json

f = open("./ZURU-ICFG-FORMAT.json")
data = json.load(f)
f.close()

for i in range(len(data)):
    data[i]['id'] += 4103 #to avoid conflicts with ICFG

f = open("../../Datasets/ICFG/ICFG-PEDES.json")
data2 = json.load(f)
f.close()

for i in range(len(data2)):
    if data2[i]['split'] == "train" and data2[i]['id'] < 500:
        data2[i]['split'] = 'val'
        data2[i]['processed_tokens'] = [data2[i]['processed_tokens'],]

s = set()
for i in range(len(data2)):
    if data2[i]['split'] == "val":
        s.add(data2[i]['id'])

print("val set of icfg val", len(s))

f = open("ICFG-PEDES-val.json", "w")
json.dump(data2, f)
f.close()

combined = data + data2

f = open("combined_datasets.json", "w")
json.dump(combined, f)
f.close()

print("SUCCESS")