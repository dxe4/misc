function partition(data, l, h)
    pivot_i = int((l + h) / 2)
    pivot = data[pivot_i]

    data[h], data[pivot_i] = data[pivot_i], data[h]
    store_i = l

    for i in l:h
        if data[i] < pivot
            data[store_i], data[i] = data[i], data[store_i]
            store_i = store_i + 1
        end
    end

    data[h], data[store_i] = data[store_i], data[h]
    return store_i
end

function qs(data, l, h)
    if l < h
        p = partition(data, l, h)
        qs(data, l, p - 1)
        qs(data, p + 1, h)
    end
end

data = rand(1:100, 16)
qs(data, 1, length(data))
print(data)
