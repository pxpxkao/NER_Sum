import os

f1 = open('input.txt', 'r')
f2 = open('task1_ref0.txt', 'r')

d1 = f1.readlines()
d2 = f2.readlines()

for i in range(len(d1)):
	if(d1[i] == "UNK\n"):
		del d1[i]
		del d2[i]
		i = i - 1
		print(i)

f1_ = open('test.article.cut.txt', 'w')
f2_ = open('test.title.cut.txt', 'w')

for i in range(len(d1)):
	f1_.write(d1[i])
	f2_.write(d2[i])