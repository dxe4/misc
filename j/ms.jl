function merge(left, right)
    result = Int64[]
    i, j = length(left), length(right)
    l, r = 1, 1

    while l <= i && r <= j
        if left[l] < right[r]
            push!(result, left[l])
            l += 1
        else
            push!(result, right[r])
            r += 1
        end
    end

    while l <= i
        push!(result, left[l])
        l += 1
    end

    while r <= j
        push!(result, right[r])
        r += 1
    end

    return result
end

function ms(data)
    if length(data) == 1
        return data
    end

    middle = int(length(data) / 2)

    left = data[1:middle]
    right = data[middle+1:length(data)]

    left = ms(left)
    right = ms(right)

    return merge(left, right)
end

data = rand(1:100, 16)
result = ms(data)
print(result)
