n = int(input())
vs = input().split(" ")
vs.insert(0, int(vs[0]))
vs.append(int(vs[-1]))

state = 'start'
current_top = int(vs[0])
values = [0]

for i in range(0, len(vs) - 1, 1):
    a, b = map(int, (vs[i], vs[i + 1]))
    diff = b - a
    if a > current_top:
        current_top = a

    elif diff >= 0:
        values.append(current_top - a)

print(-max(values))
