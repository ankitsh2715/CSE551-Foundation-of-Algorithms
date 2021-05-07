import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

answer = []
inputPath = os.path.join(os.curdir, r"input.txt")

def contructFlowGraph(dim, vList):   
    # number of nodes after splitting each vertex in Vin and Vout, and adding two new nodes for source and sink
    numNodes = dim * dim * 2 + 2
    network = np.zeros((numNodes, numNodes), dtype=int)

    # Vin-Vout edges changed to 1
    for i in range(1, dim * dim * 2, 2):
        network[i][i + 1] = 1

    for i in range(1, dim + 1):
        network[2 * i][numNodes - 1] = 1
        network[2 * (dim - 1) * dim + 2 * i][numNodes - 1] = 1

    for i in range(dim):
        network[i * 2 * dim + 2][numNodes - 1] = 1
        network[2 * dim * (i + 1)][numNodes - 1] = 1

    # add connection from Source (S) to Vertex(Vin)
    for v in vList:
        network[0][(v[0] - 1) * 2 * dim + (2 * (v[1] - 1)) + 1] = 1

    # add connection from Vertex to its right neighbour
    for v in range(2, dim * dim * 2 + 1, 2):
        if v % (2 * dim) != 0:
            network[v][v + 1] = 1

    # add connection from Vertex to its left neighbour
    for v in range(2, dim * dim * 2 + 1, 2):
        if (v - 2) % (2 * dim) != 0:
            network[v][v - 3] = 1

    # add connection from Vertex to its top neighbour
    for v in range(2, dim * dim * 2 + 1, 2):
        if v - (2 * dim) > 0:
            network[v][v - (2 * dim) - 1] = 1

    # add connection from Vertex to its down neighbour
    for v in range(2, dim * dim * 2 + 1, 2):
        if v + (2 * dim) < dim * dim * 2 + 1:
            network[v][v + (2 * dim) - 1] = 1

    maxFlow = edmondsKarp(network, 0, numNodes - 1, dim)

    return maxFlow



def edmondsKarp(graph, s, t, dim):
    flow = 0
    source, sink = s, t

    temp1, predecessor = breadth_first_order(csr_matrix(graph), 0, directed=True, return_predecessors=True)
    sPath = shortestPath(predecessor, source, sink)

    while source in sPath:
        minCap = findMinEdgeCap(graph, sPath)
        flow = flow + minCap
        graph = augmentPath(graph, sPath, minCap)
        temp2, predecessor = breadth_first_order(csr_matrix(graph), 0, directed=True, return_predecessors=True)
        
        #store escape route to print
        escapeRoute=" "
        for i in range(1,len(sPath)-1,2):
            for x in range(1,dim+1):
                for y in range(1,dim+1):
                    if(((x-1)*2*dim + (2*(y-1)+1))==sPath[i]):
                        escapeRoute = escapeRoute + "(" + str(x) + "," + str(y) + ")" + " -> "
                        break
        
        if escapeRoute!=" ":
            answer.append(escapeRoute[:-4])
        
        sPath = shortestPath(predecessor, source, sink)

    return flow


def shortestPath(arr, s, t):
    temp = [t]
    i = t
    while arr[i] != -9999:
        temp.append(arr[i])
        i = arr[i]

    return temp[::-1]


def findMinEdgeCap(graph, arr): 
    minCap = np.inf
    for i in range(1, len(arr)):
        start, end = arr[i - 1], arr[i]
        if graph[start][end] < minCap:
            minCap = graph[start][end]

    return minCap


def augmentPath(resGraph, sPath, minEdge): 
    for i in range(1, len(sPath)):
        start, end = sPath[i - 1], sPath[i]
        resGraph[start][end] -= minEdge
        resGraph[end][start] += minEdge

    return resGraph

if __name__ == '__main__':

    inputList = []
    
    with open(inputPath) as file:
        for line in file:
            inputList.append((map(int, line.strip().split())))

    numVertices, n = inputList[0]
    startNodeList = []
    for i in range(numVertices):
        x, y = inputList[i + 1]
        startNodeList.append((x, y))

    maxFlow = contructFlowGraph(n, startNodeList)
    if maxFlow == len(startNodeList):
        print("\n(i) YES, a solution exists.")
        print("(ii) A solution to this problem is:")
        for ans in answer:
            pos = ans.find(")")+1
            print("\tPATH from " + ans[:pos] + ":  " + ans)
    else:
        print("\nNO solution exists\n")
