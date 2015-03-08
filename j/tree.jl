abstract AbstractNode
type EmptyNodeType <: AbstractNode end
const EmptyNode = EmptyNodeType()


type Node <: AbstractNode
    left::AbstractNode
    right::AbstractNode
    parent::AbstractNode
    value
end


function make_btree(depth)
#                      0
#             1                  2
#        10       11       20        21
#     100 101  110 111  200 201    210 211
#                   etc
    root = Node(EmptyNode, EmptyNode, EmptyNode, 0)
    current_nodes = [root]
    depth = 10

    for level=1:depth
        next_nodes = AbstractNode[]

        for node in current_nodes
            new_value = node.value * 10
            if new_value == 0
                new_value = 1
            end

            left = Node(EmptyNode, EmptyNode, node, new_value)
            right = Node(EmptyNode, EmptyNode, node, new_value + 1)

            node.left = left
            node.right = right

            push!(next_nodes, left, right)
        end

        current_nodes = next_nodes
    end
end
