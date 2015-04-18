import math
from collections import defaultdict
from random import randint

data = [(i, randint(100, 130)) for i in range(0, 50)]

x = [i[1] for i in data]
y = [i[0] for i in data]

start, end = min(x), max(x)
mean = sum(x) / len(x)


differences = [(i - mean) ** 2 for i in x]
variance = sum(differences) / len(differences)

standard_deviation = math.sqrt(variance)
# scotts rule
bucket_size = int(3.49 * standard_deviation / math.log(len(x), 3))

buckets = [i for i in range(start, end + bucket_size, bucket_size)]

hist = defaultdict(int)

for value in x:
    start_i = 0
    while start_i < len(buckets) - 1:
        if value >= buckets[start_i] and value <= buckets[start_i + 1]:
            hist[start_i] += 1
            break
        start_i += 1

print(dict(hist))
print(buckets)
