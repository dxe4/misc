import sys
import math
from copy import copy
from collections import defaultdict

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# N: the total number of nodes in the level, including the gateways
# L: the number of links
# E: the number of exit gateways
N, L, E = [int(i) for i in input().split()]
links = defaultdict(list)

for i in range(L):
    # N1: N1 and N2 defines a link between these nodes
    N1, N2 = [int(i) for i in input().split()]
    links[N1].append(N2)

gateways = []
for i in range(E):
    EI = int(input())  # the index of a gateway node
    gateways.append(EI)


def traverse(graph, current_path, start, end_list, result):
    '''
    Return a dict from score to paths
    Key of the dict is the lenght of the path (to allow picking shortest)
    Value of the dict is a list of paths
    '''
    if start in current_path:
        # Avoid infinite recursion
        return

    current_path = copy(current_path)
    current_path.append(start)

    if start in end_list:
        result[len(current_path)].append(current_path)

    try:
        next_nodes = graph[start]
    except KeyError:  # End
        return current_path

    for node in next_nodes:
        traverse(graph, current_path, node, end_list, result)

# game loop
while 1:
    # The index of the node on which the Skynet agent is positioned this turn
    SI = int(input())
    result = defaultdict(list)
    traverse(links, [], SI, gateways, result)
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)

    # TODO clean this up
    keys = sorted(result.keys())
    if keys:
        link = result[keys[0]][0]
        # Example: 0 1 are the indices of the nodes you wish to sever the link
        # between
        print(link[0], link[1])
    else:
        print('WAIT')
