abstract AbstractNode
type EmptyNodeType <: AbstractNode end

type Edge
    node::AbstractNode
    cost::Int64
end

type Node <: AbstractNode
    parent::AbstractNode
    edges::Array{Edge, 1}
    value
end

const EmptyNode = EmptyNodeType()
const EmptyEdges = Edge[]

function default_get(arr, index, default=0)
    if arr == nothing || (length(arr) < index)
        return 0
    else
        return getindex(arr, index)
    end
end


function edges_array(node::Node)
    if length(node.edges) == 0
        # Because of the const EmptyEdges, probably will cause trouble
        node.edges = Edge[]
    end

    return node.edges
end


function add_edge(node, edge::Edge)
    edges = edges_array(node)
    push!(edges, edge)
end


function add_edges(node, new_edges::Array{Edge})
    edges = edges_array(node)

    for edge in new_edges
        push!(edges, edge)
    end
end


function add_edges(node, node_values::Array{ASCIIString},
                   scores::Array{Int64}=nothing)
    edges = edges_array(node)

    for (index, value) in enumerate(node_values)
        node = Node(node, EmptyEdges, value)
        score = default_get(scores, index)
        edge = Edge(node, score)

        push!(edges, edge)
    end
end


function add_nodes(node, new_nodes::Array{Node},
                   scores::Array{Int64}=nothing)
    edges = edges_array(node)

    for node in new_nodes
        score = default_get(scores, index)
        edge = Edge(node, score)
        push!(edges, edge)
    end
end


function make_graph()
    root = Node(EmptyNode, EmptyEdges, "A")

    node_values = ["A => " * i for i in ["B", "C", "D", "E"]]
    scores = [index for (index, node) in enumerate(node_values)]

    add_edges(root, node_values, scores)

    for i in root.edges
        print(i.node.value, ", ")
    end
end
