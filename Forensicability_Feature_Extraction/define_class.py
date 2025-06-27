import csv
import math

e_c_l = [[1.0, 1.0], [1.0, 0.0], [0.0, 0.5]]

def L2_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

csv_result = {}
with open("data/train.csv") as f:
    f_csv = csv.reader(f)

    headers = next(f_csv)
    for idx, row in enumerate(f_csv):
        csv_result[str(idx)] = [float(row[-3]), float(row[-2])]

e3_distance_l = []
for item in csv_result:
    e3_distance_l.append(L2_distance(csv_result[item], e_c_l[2]))

e3_distance_l.sort()
e3_threshold = e3_distance_l[int(len(e3_distance_l) * 0.3)]

class_l = []
for item in csv_result:
    if L2_distance(csv_result[item], e_c_l[2]) < e3_threshold:
        class_l.append("2")
    elif L2_distance(csv_result[item], e_c_l[1]) < L2_distance(csv_result[item], e_c_l[0]):
        class_l.append("1")
    else:
        class_l.append("0")

with open("data/class.csv", "w+", newline="") as f:
    f_csv = csv.writer(f)
    f_csv.writerows(class_l)