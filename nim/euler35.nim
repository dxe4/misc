import sets
import strutils
import sequtils

import times, os, strutils

template benchmark(benchmarkName: string, code: untyped) =
  block:
    let t0 = epochTime()
    code
    let elapsed = epochTime() - t0
    let elapsedStr = elapsed.formatFloat(format = ffDecimal, precision = 3)
    echo "CPU Time [", benchmarkName, "] ", elapsedStr, "s"

benchmark "my benchmark":

    let input = open("primes.txt")
    var primes: HashSet[string]
    init(primes)

    for line in input.lines:
        primes.incl(line)


    var cnt: int = 0

    for prime in primes:
        var t: int = len(prime)
        var isgood: bool = true
        for i in 0..t:
            var tstr: string = substr(prime, i) & substr(prime, 0, i-1)
            if not primes.contains(tstr):
                isgood = false
                break

        if isgood:
            cnt += 1
`
    echo cnt
