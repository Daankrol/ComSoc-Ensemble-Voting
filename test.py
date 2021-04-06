from voting_rules import *

c1 = [[x for x in range(26)]]
c2 = [[x for x in range(26)]]
c3 = [[x for x in range(26)]]
c4 = [[x for x in range(26)]]
r = [c1, c2, c3, c4]

c2[0][0] = 1
c2[0][1] = 2
c2[0][2] = 0

c3[0][0] = 2
c3[0][1] = 1
c3[0][2] = 0

c4[0][0] = 3
c4[0][1] = 1
c4[0][2] = 2
c4[0][3] = 0

print(c1, '\n', c2, '\n')

print(r)
result = plurality(r)
print(result)
print('!!!')
print(STV(r))

c1 = [[x for x in range(26)]]
c2 = [[x for x in range(26)]]
c3 = [[x for x in range(26)]]

c1[0][0] = 2
c1[0][1] = 1
c1[0][2] = 0

c2[0][0] = 1
c2[0][1] = 0
c2[0][2] = 2

r = [c1, c2, c3]
print(plurality(r))
print(STV(r))
