// Read inputs from System.in, Write outputs to System.out.

input = new Scanner(System.in)
n = input.nextInt()

values = []
min = 10000001
for (i in 1..n ) {
    a = input.nextInt()
    values.add(a)
}
values = values.sort()

for (int i = 0; i < n-1; i=i+2) {
    temp = values[i+1] - values[i]
    if (temp < min && temp != 0) {
        min = temp
    }
}
// e.g. if 5 items we need to compare 4 and 5
if (n % 2 != 0) {
    temp = values[n-1] - values[n-2]
    if (temp < min && temp != 0) {
        min = temp
    }
}

if(min == 10000001){
    min = 0
}

println min
