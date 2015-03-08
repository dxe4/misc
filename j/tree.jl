abstract AbstractNode
type EmptyNodeType <: AbstractNode end
const EmptyNode = EmptyNodeType()


type BNode <: AbstractNode
    left::AbstractNode
    right::AbstractNode
    parent::AbstractNode
    value
end


function make_btree()
#                      A
#             B                  C
#        D       E           F        G
#     H    I   J   K       L   M    N   O
#                   etc

    root = BNode(EmptyNode, EmptyNode, EmptyNode, "A")
    current_nodes = [root]
    items = [
        "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"
    ]

    index = 1
    while index <= length(items)
        next_nodes = AbstractNode[]

        for node in current_nodes
            left = BNode(EmptyNode, EmptyNode, node, items[index])
            right = BNode(EmptyNode, EmptyNode, node, items[index + 1])

            node.left = left
            node.right = right

            push!(next_nodes, left, right)
            index += 2
        end

        current_nodes = next_nodes
    end
    return root
end


function pre_order(node, result)
    if typeof(node) == EmptyNodeType
        return
    end

    push!(result, node)
    pre_order(node.left, result) 
    pre_order(node.right, result)

    return result
end

function in_order(node, result)
    if typeof(node) == EmptyNodeType
        return
    end

    pre_order(node.left, result) 
    push!(result, node)
    pre_order(node.right, result)

    return result
end

function post_order(node, result)
    if typeof(node) == EmptyNodeType
        return
    end

    pre_order(node.left, result) 
    pre_order(node.right, result)
    push!(result, node)

    return result
end

function bfs(root, result)
    queue = AbstractNode[]
    push!(queue, root)

    while length(queue) > 0

        node = getindex(queue, 1)
        deleteat!(queue, 1)  # Not ideal
        push!(result, node)

        if typeof(node.left) != EmptyNodeType
            push!(queue, node.left)
        end
        if typeof(node.right) != EmptyNodeType
            push!(queue, node.right)
        end
    end
    return result
end


function _print(result, func_type)
    print(func_type, ": ")
    print([i.value for i in result])
    println()
end

root = make_btree()

result = bfs(root, AbstractNode[])
_print(result, "bfs")

result = pre_order(root, AbstractNode[])
_print(result, "pre_order")

result = post_order(root, AbstractNode[])
_print(result, "post_order")

result = in_order(root, AbstractNode[])
_print(result, "in_order")
