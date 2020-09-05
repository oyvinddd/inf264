import numpy as np

class Node:
    def __init__(self, name, children=[]):
        self.name = name
        self.children = children
    def add_child(self, node):
        self.children.append(node)

# build a tree from a flat array
def build_tree(l, node):
    if len(l) == 1:
        node.add_child(Node(str(l[0]), []))
        return
    child_a = Node('internal', [])
    child_b = Node('internal', [])
    node.add_child(child_a)
    node.add_child(child_b)
    build_tree(l[:len(l)//2], child_a)
    build_tree(l[len(l)//2:], child_b)

def print_tree(tree, indent, last):
    print(indent + '+- ' + tree.name)
    if last:
        indent += "   "
    else:
        indent += "|  "
    for idx, val in enumerate(tree.children):
        print_tree(tree.children[idx], indent, len(tree.children)-1 == idx)

tree = Node('root', [])
build_tree([1,2,3,4,5,6], tree)
print_tree(tree, '', True)