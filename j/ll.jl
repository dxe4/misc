type N
    next
    value
end

type LL
    start::N
end

function printll(ll::LL)
    println()
    current = ll.start

    while current != nothing && current.next != nothing
        print(current.value, ", ")
        current = current.next
    end

    print(current.value)
    println()
end

function initialize()
    current = N(nothing, 0)
    ll = LL(current)

    for elm in 1:15
        n = N(nothing, elm)
        current.next = n
        current = n
    end

    return ll
end

function reverse(ll::LL)
    current = ll.start
    next = current.next
    current.next = nothing

    while next != nothing
        temp = next.next
        next.next = current
        current = next
        next = temp
    end

    ll.start = current
end

function ll_middle(ll::LL)
    i = 0
    current = ll.start
    m = current

    while current != nothing
        if i % 2 == 0
            m = m.next
        end
        current = current.next
        i += 1
    end

    return m
end

ll = initialize()

printll(ll)

reverse(ll)

printll(ll)

println()
print(ll_middle(ll).value)
