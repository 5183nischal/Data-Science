import csv
with open('6_clique_sol', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

lst = your_list[0]

s = 0
for i in range(len(lst)):
	s = s + int(lst[i])

print(s)