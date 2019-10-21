from pathlib import Path
import csv
import json

TESTPATH = Path.cwd() / "prediction.csv"
CONFIG = Path.cwd() / "config.json"


def write():
    fn = open(str(TESTPATH), "r", encoding="utf-8")
    ret = csv.reader(fn, delimiter=',')
    data = []
    for x in ret:
        data.append(x)
    with open(CONFIG) as json_file:
        target = json.load(json_file)
    for i in data:
        for key in target:
            if str(i[1]) == str(target[key]):
                i[1] = key
    writer = csv.writer(open("result.csv", 'w'))
    for row in data:
        writer.writerow(row)
