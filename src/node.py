class Node:
    def __init__(self, node_type, value, left=None, right=None):
        """
        Parameters:
            node_type: 'binary_op', 'unary_op', 'variable', or 'constant'
            value: For operators, a function from the provided lists.
                For 'variable', typically a string (e.g. 'x').
                For 'constant', a numeric value.
            left, right: Child nodes (if applicable)
        """
        self.node_type = node_type
        self.value = value
        self.left = left    # For binary and unary operators (left child holds the operand)
        self.right = right  # For binary operators

    def clone_node(self):
        """
        Create a deep copy of the node and its subtree.
        
        Returns:
            Node: A new node that is a deep copy of the current node and its subtree.
        """
        left_clone = self.left.clone_node() if self.left is not None else None
        right_clone = self.right.clone_node() if self.right is not None else None
        return Node(self.node_type, self.value, left_clone, right_clone)

    def to_string(self):
        """
        Convert the node and its subtree to a string representation.
        
        Returns:
            str: The string representation of the node and its subtree.
        """
        if self.value is None:
            return None
        if self.node_type == 'constant':
            return str(self.value)
        if self.node_type == 'variable':
            return "x[" + self.value[1:] + "]"
        if self.node_type == 'unary_op':
            op = self.right.to_string() if self.left is None else self.left.to_string()
            return f"np.{self.value.__name__}({op})"
        if self.node_type == 'binary_op':
            op_left = self.left.to_string()
            op_right = self.right.to_string()
            return f"np.{self.value.__name__}({op_left}, {op_right})"