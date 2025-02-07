import numpy as np
import random
import matplotlib.pyplot as plt
from node import Node

# List of unary operations
unary_ops = [
    np.negative,
    np.abs,
    np.sqrt,
    np.exp,
    np.log,
    np.log2,
    np.log10,
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.square,
    np.cbrt,
    np.reciprocal,
]

# List of binary operations
binary_ops = [
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.power,
    np.maximum,
    np.minimum,
    np.mod
]

class Tree:
    def __init__(self, max_depth, x_train, y_train, tree_attempts):
        """
        Initialize the Tree object.
        
        Parameters:
            max_depth (int): Maximum allowed depth for the tree.
            x_train (np.ndarray): Training data for the input variables.
            y_train (np.ndarray): Training data for the target values.
            tree_attempts (int): Number of attempts to find a valid mutation or recombination.
        """
        self.root = None
        self.max_depth = max_depth
        self.fitness = None  # Will hold the computed Mean Squared Error (MSE)
        self.x_train = x_train
        self.y_train = y_train
        self.var_num = x_train.shape[0]
        self.vars = [f'x{i}' for i in range(self.var_num)]
        self.tree_attempts = tree_attempts
        
    def populate(self):
        """Generate a random tree and compute its fitness using MSE."""
        self.root = self.generate_random_tree()
        self.compute_fitness()

    def generate_random_tree(self, current_depth=0, max_depth=None):
        """
        Recursively generate a random expression tree.
        At maximum depth, only terminal nodes (variable or constant) are allowed.
        
        Parameters:
            current_depth: The current depth in the tree.
            max_depth: Maximum allowed depth for this subtree. If None, use self.max_depth.
        
        Returns:
            Node: The root node of the generated subtree.
        """
        if max_depth is None:
            max_depth = self.max_depth

        # Terminal node if max allowed depth reached.
        if current_depth >= max_depth:
            if random.random() < 0.5:
                return Node('variable', self.vars[random.randint(0, self.var_num - 1)])
            else:
                return Node('constant', random.uniform(-10, 10))
        else:
            # With higher probability choose an operator node.
            if random.random() < 0.7:
                # Randomly decide between binary and unary operator.
                if random.random() < 0.5:
                    # Binary operator node.
                    op = random.choice(binary_ops)
                    left = self.generate_random_tree(current_depth + 1, max_depth)
                    right = self.generate_random_tree(current_depth + 1, max_depth)
                    return Node('binary_op', op, left, right)
                else:
                    # Unary operator node.
                    op = random.choice(unary_ops)
                    child = self.generate_random_tree(current_depth + 1, max_depth)
                    return Node('unary_op', op, child, None)
            else:
                # Terminal node.
                if random.random() < 0.5:
                    return Node('variable', self.vars[random.randint(0, self.var_num - 1)])
                else:
                    return Node('constant', random.uniform(-10, 10))

    def compute_fitness(self):
        """Compute the fitness of the tree using Mean Squared Error (MSE)."""
        x_data = self.x_train
        y_data = self.y_train
        # Convert the formula to a pre-compiled lambda function
        formula = self.root.to_string()  
        eval_formula = eval(f"lambda x: {formula}", {"np": np, "nan": np.nan, "inf": np.inf}) 

        # Exploiting np broadcasting
        y_pred = eval_formula(x_data)  

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            self.fitness = np.inf
            return

        # Broadcasting is used to calculate the squared errors
        squared_errors = np.square(y_data - y_pred)
        self.fitness = np.sum(squared_errors) / x_data.shape[1]

    def copy(self):
        """
        Return a deep copy of the tree.
        
        Returns:
            Tree: A new tree object that is a deep copy of the current tree.
        """
        new_tree = Tree(max_depth=self.max_depth, x_train=self.x_train, y_train=self.y_train, tree_attempts=self.tree_attempts)
        new_tree.root = self.root.clone() if self.root is not None else None
        new_tree.fitness = self.fitness
        return new_tree

    def get_all_nodes(self, node=None, parent=None, is_left=None, depth=0):
        """
        Recursively traverse the tree and return a list of tuples:
        (node, parent, is_left, depth) where depth is the node's depth from the root.
        
        Parameters:
            node: The current node being traversed.
            parent: The parent of the current node.
            is_left: Boolean indicating if the current node is a left child.
            depth: The depth of the current node from the root.
        
        Returns:
            list: A list of tuples (node, parent, is_left, depth).
        """
        if node is None:
            node = self.root
        nodes = [(node, parent, is_left, depth)]
        if node.left is not None:
            nodes.extend(self.get_all_nodes(node.left, node, True, depth + 1))
        if node.right is not None:
            nodes.extend(self.get_all_nodes(node.right, node, False, depth + 1))
        return nodes

    def _get_subtree_height(self, node):
        """
        Compute the height of the subtree rooted at the given node.
        A leaf node has height 1.
        
        Parameters:
            node: The root node of the subtree.
        
        Returns:
            int: The height of the subtree.
        """
        if node is None:
            return 0
        left_height = self._get_subtree_height(node.left) if node.left is not None else 0
        right_height = self._get_subtree_height(node.right) if node.right is not None else 0
        return 1 + max(left_height, right_height)

    def mutate(self):
        """Perform mutation on the tree."""
        if random.random() < 0.6:
            self.mutate_subtree()
        else:
            mutate_choice = random.choice([0, 1, 2, 3])
            if mutate_choice == 0:
                self.mutate_point()
            elif mutate_choice == 1:
                self.mutate_permutation()
            elif mutate_choice == 2:
                self.mutate_hoist()
            else:
                self.mutate_collapse()

    def mutate_subtree(self):
        """
        Perform mutation by selecting a random node in the tree and replacing it
        with a new randomly generated subtree that does not cause the overall
        tree depth to exceed the allowed maximum. Keep creating a new subtree
        until the fitness is valid.
        """
        nodes = self.get_all_nodes()  # Each tuple is (node, parent, is_left, depth)
        attempts = self.tree_attempts
        valid_subtree_found = False
        
        while attempts > 0 and not valid_subtree_found:
            node_to_mutate, parent, is_left, node_depth = random.choice(nodes)
            
            # Calculate remaining depth allowed for the new subtree.
            remaining_depth = self.max_depth - node_depth
            
            # Generate a new subtree with maximum allowed height equal to the remaining depth.
            new_subtree = self.generate_random_tree(current_depth=0, max_depth=remaining_depth)
            
            # Temporarily replace the selected node (or the entire tree if it is the root).
            if parent is None:
                self.root = new_subtree
            else:
                if is_left:
                    parent.left = new_subtree
                else:
                    parent.right = new_subtree
            
            # Compute fitness and check if it is valid.
            self.compute_fitness()
            if self.fitness != np.inf:
                valid_subtree_found = True
            else:
                # Revert the change if the fitness is not valid.
                if parent is None:
                    self.root = node_to_mutate
                else:
                    if is_left:
                        parent.left = node_to_mutate
                    else:
                        parent.right = node_to_mutate
            
            attempts -= 1

    def mutate_point(self):
        """Performs a point mutation by changing a randomly selected node."""
        nodes = self.get_all_nodes()
        if not nodes:
            return
        node, _, _, _ = random.choice(nodes)
        
        if node.node_type == 'binary_op':
            node.value = random.choice(binary_ops)
        elif node.node_type == 'unary_op':
            node.value = random.choice(unary_ops)
        
        self.compute_fitness()
    
    def mutate_permutation(self):
        """Swaps the left and right child of a randomly chosen binary operator."""
        binary_nodes = [n for n in self.get_all_nodes() if n[0].node_type == 'binary_op']
        if not binary_nodes:
            return
        node, _, _, _ = random.choice(binary_nodes)
        node.left, node.right = node.right, node.left
        self.compute_fitness()
    
    def mutate_hoist(self):
        """Replaces a randomly selected subtree with one of its sub-subtrees."""
        nodes = self.get_all_nodes()
        attempts = self.tree_attempts
        if not nodes:
            return
        
        node, _, _, _ = random.choice(nodes)
        while attempts > 0 and node.left is None and node.right is None:
            attempts -= 1
            node, _, _, _ = random.choice(nodes)
            
        if attempts <= 0:
            return
        self.root = node
        self.compute_fitness()
    
    def mutate_collapse(self):
        """Replaces a randomly selected subtree with a constant equal to its evaluated mean value."""
        nodes = self.get_all_nodes()
        if not nodes:
            return

        attempts = self.tree_attempts
        valid_collapse_found = False

        while attempts > 0 and not valid_collapse_found:
            node, parent, is_left, _ = random.choice(nodes)

            if node.node_type not in ['binary_op', 'unary_op']:
                attempts -= 1
                continue  # Skip if it's not a valid operator node

            try:
                formula = node.to_string()  # Get the string formula
                eval_formula = eval(f"lambda x: {formula}", {"np": np, "nan": np.nan, "inf": np.inf})
                collapsed_value = float(np.mean(eval_formula(self.x_train)))
            except:
                collapsed_value = 0.0

            new_node = Node('constant', collapsed_value)
            if parent is None:
                self.root = new_node
            else:
                if is_left:
                    parent.left = new_node
                else:
                    parent.right = new_node

            self.compute_fitness()

            if self.fitness != np.inf:
                valid_collapse_found = True
            else:
                # Revert the change if fitness is invalid
                if parent is None:
                    self.root = node
                else:
                    if is_left:
                        parent.left = node
                    else:
                        parent.right = node

            attempts -= 1

    def recombine(self, other_tree):
        """
        Perform recombine with another tree by swapping randomly chosen subtrees.
        The swap is only accepted if it does not cause the overall depth of either
        offspring to exceed the maximum allowed depth and the fitness is valid.
        
        Parameters:
            other_tree: The other tree to recombine with.
        
        Returns:
            tuple: Two new offspring trees.
        """
        offspring1 = self.copy()
        offspring2 = other_tree.copy()
        
        nodes1 = offspring1.get_all_nodes()  # (node, parent, is_left, depth)
        nodes2 = offspring2.get_all_nodes()
        
        # We'll try a fixed number of attempts to find two subtrees that fit.
        attempts = self.tree_attempts
        valid_swap_found = False
        
        while attempts > 0 and not valid_swap_found:
            node1, parent1, is_left1, depth1 = random.choice(nodes1)
            node2, parent2, is_left2, depth2 = random.choice(nodes2)
            
            height1 = offspring1._get_subtree_height(node1)
            height2 = offspring2._get_subtree_height(node2)
            
            # Check: after swapping, the new depth in offspring1 would be:
            #   depth1 (depth of the replacing location) + height2 (height of subtree from tree2)
            # Similarly for offspring2.
            if (depth1 + height2 <= self.max_depth) and (depth2 + height1 <= self.max_depth):
                # Clone the chosen subtrees.
                subtree1 = node1.clone()
                subtree2 = node2.clone()
                
                # Swap subtree in offspring1.
                if parent1 is None:
                    offspring1.root = subtree2
                else:
                    if is_left1:
                        parent1.left = subtree2
                    else:
                        parent1.right = subtree2
                        
                # Swap subtree in offspring2.
                if parent2 is None:
                    offspring2.root = subtree1
                else:
                    if is_left2:
                        parent2.left = subtree1
                    else:
                        parent2.right = subtree1

                # Compute fitness and check if it is valid.
                offspring1.compute_fitness()
                offspring2.compute_fitness()
                
                if offspring1.fitness != np.inf and offspring2.fitness != np.inf:
                    valid_swap_found = True
                    break
                else:
                    # Revert the change if the fitness is not valid.
                    if parent1 is None:
                        offspring1.root = node1
                    else:
                        if is_left1:
                            parent1.left = node1
                        else:
                            parent1.right = node1
                            
                    if parent2 is None:
                        offspring2.root = node2
                    else:
                        if is_left2:
                            parent2.left = node2
                        else:
                            parent2.right = node2

            attempts -= 1
        
        # If no valid swap was found, return the unchanged offspring.
        if not valid_swap_found:
            return self.copy(), other_tree.copy()
        
        return offspring1, offspring2

    def plot(self):
        """Draws the tree using matplotlib. If a node has only one child, the line to that child is drawn vertically.
        The vertical gap between parent and child is minimized by drawing the line from center to center."""
        def draw_node(node, x, y, dx, dy):
            if node is not None:
                # Determine the text to display for the node.
                if node.node_type in ["binary_op", "unary_op"]:
                    text = node.value.__name__
                elif node.node_type == "variable":
                    text = node.to_string()
                elif node.node_type == "constant":
                    # Convert the constant to a float and round it.
                    text = str(round(float(node.to_string()), 2))
                else:
                    text = ""
                
                # Set color: variables are red; constants are lightgreen; others are lightblue.
                if node.node_type == "variable":
                    color = 'tomato'
                elif node.node_type == "constant":
                    color = 'lightgreen'
                else:
                    color = "lightblue"
                
                # Draw the node text at its (x, y) position.
                plt.text(x, y, text, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=color))
                
                # Draw the connection lines and recursively draw child nodes.
                if node.left is not None and node.right is not None:
                    # Two children: draw left and right branches with horizontal offsets.
                    plt.plot([x, x - dx], [y, y - dy], color='black')  # line to left child
                    draw_node(node.left, x - dx, y - dy, dx / 2, dy)
                    plt.plot([x, x + dx], [y, y - dy], color='black')  # line to right child
                    draw_node(node.right, x + dx, y - dy, dx / 2, dy)
                elif node.left is not None:
                    # Only left child exists: draw a vertical line.
                    plt.plot([x, x], [y, y - dy], color='black')
                    draw_node(node.left, x, y - dy, dx / 2, dy)
                elif node.right is not None:
                    # Only right child exists: draw a vertical line.
                    plt.plot([x, x], [y, y - dy], color='black')
                    draw_node(node.right, x, y - dy, dx / 2, dy)

        plt.figure(figsize=(15, 10))
        plt.axis('off')
        draw_node(self.root, 0, 0, 20, 1)
        plt.show()

    def __str__(self):
        """
        Return the string representation of the tree (its formula).
        
        Returns:
            str: The string representation of the tree's formula.
        """
        return self.root.to_string() if self.root is not None else ""