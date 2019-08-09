from collections import defaultdict
from random import choice

class FindCycle:
    #finds the big cycle in connected undirected graphs with one big cycle

    def __init__(self):
        self._neighbors = defaultdict(list)
        self._parent = defaultdict()
        self.cycle = []
        self.tree = []
    
    def addEdge(self, u, v):
        if u != v:
            if v not in self._neighbors[u]:
                self._neighbors[u].append(v)
            if u not in self._neighbors[v]:
                self._neighbors[v].append(u)
    
    def findCycle(self):
        visited = dict.fromkeys(self._neighbors, False)

        s = choice(list(visited.keys()))

        stack = [s]
        while len(stack):
            v = stack.pop(0)
            if not visited[v]:
                visited[v] = True
                for u in self._neighbors[v]:
                    if not visited[u]:
                        self._parent[u] = v
                        stack.insert(0, u)
                    elif v != s and u != self._parent[v]:
                        cycle = [u]
                        w = v
                        while w != u:
                            cycle.append(w)
                            w = self._parent[w]
                        if len(cycle) > len(self.cycle):
                            self.cycle = cycle