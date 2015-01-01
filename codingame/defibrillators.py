import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

_float = lambda n: float(n.replace(",", "."))
LON = _float(input())
LAT = _float(input())
N = int(input())
rows = {}
names = {}

for i in range(N):
    DEFIB = input()

    try:
        head, tail = DEFIB.split(";;")
    except ValueError as e:
        defib = DEFIB.split(";")
        head = ";".join(defib[0:-2])
        tail = ";".join(defib[-2:])

    id, name = head.split(";")[0:2]
    names[id] = name

    lonlat = map(_float, tail.split(";"))
    rows[id] = lonlat

distances = []
for k, v in rows.items():
    lon, lat = v
    x = (LON - lon) * math.cos((LAT + lat) / 2)
    y = LAT - lat
    d = math.sqrt(x**2 + y**2) * 6371
    distances.append((k, d))


distances = sorted(distances, key=lambda n: n[1])

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr)

id = distances[0][0]
name = names[id]
print(name)
