import itertools

H = ['S', 'M', 'L']
CAPH = {'S':500, 'M':1000, 'L':1500}
EM = {'S':20, 'M':25, 'L':30} # test different shapes
NF = 10
NC = 10
F = list(range (1, NF +1 ))
C = list(range (NF+1, NC + NF +1 ))

c_cap = {(j,h): CAPH[h] for (j, h) in itertools.product(F,H)}
c_cost = {(j,h): 1 for (j, h) in itertools.product(F,H)}
em = {(j,h):EM[h] for (j,h) in itertools.product(F,H)}

