class WeightedQuickUnionUF:
    """
    Weighted Quick Union UF is a Python conversion of the algorithm as implemented by Kevin Wayne and Robert Sedgewick
    for their Algorithms 1 course on Coursera.

    https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/WeightedQuickUnionUF.html
    """

    def __init__(self, n: int, entries):
        """
        Initializes an empty union–find data structure with n sites
        0 through n-1. Each site is initially in its own
        component.

        Within the parent list, each entry is a tuple that contains the "parent"
        index, as well as the object holding that specific entry (generic)
        """
        self._parent = [(0, None)] * n
        self._size = [1] * n
        self._count = n
        for i in range(n):
            self._parent[i] = (i, entries[i])

    def count(self):
        """
        Returns the number of components.
        """
        return self._count

    def find(self, p: int):
        """
        Returns the component identifier for the component containing site p.
        """
        self._validate(p)
        while p != self._parent[p][0]:
            # path compression (make every other node in path point to its grandparent)
            self._parent[p] = (self._parent[self._parent[p][0]][0], self._parent[p][1])

            p = self._parent[p][0]

        return p

    def _validate(self, p: int):
        """
        Validate that p is a valid index.
        """
        n = len(self._parent)
        if p is None or p < 0 or p >= n:
            raise ValueError("index {0} is not between 0 and {1}".format(p, n - 1))

    def connected(self, p: int, q: int):
        """
        Returns true if the the two sites are in the same component.
        """
        return self.find(p) == self.find(q)
        pass

    def union(self, p: int, q: int):
        """
        Merges the component containing site p with the component containing site q.
        """
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return

        # make smaller root point to larger one
        if self._size[root_p] < self._size[root_q]:
            self._parent[root_p] = (root_q, self._parent[root_p][1])
            self._size[root_q] = self._size[root_q] + self._size[root_p]
        else:
            self._parent[root_q] = (root_p, self._parent[root_q][1])
            self._size[root_p] = self._size[root_p] + self._size[root_q]

        self._count = self._count - 1
        pass

    def get_unions(self):
        """
        Retrieves and returns all groups, according to their parent
        """
        components = {}
        for index_element in range(len(self._parent)):
            # get parent component
            index_parent = self.find(index_element)
            parent = self._parent[index_parent][1]
            child = self._parent[index_element][1]

            # add it to the mapping, or add the current component to its list
            if index_parent not in components:
                if index_element == index_parent:
                    components[index_parent] = [
                        child,
                    ]
                else:
                    components[index_parent] = [
                        parent,
                        child,
                    ]
            else:
                parent_list = components[index_parent]
                parent_list.append(child)
                components[index_parent] = parent_list

        return components
