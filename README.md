# Symbolic Regression with Genetic Programming

## Collaboration

This project was developed with my colleague [Gabry323387](https://github.com/Gabry323387). Together, we designed and implemented the symbolic regression framework, applying genetic programming techniques to evolve mathematical expressions. Through joint experimentation and iterative improvements, we achieved accurate and interpretable results.

## Overview

This project implements a symbolic regression framework using Genetic Programming (GP), an evolutionary computation technique that generates and optimizes mathematical expressions in a tree-based structure. The goal of this project is to develop a system that can approximate a given dataset through evolved symbolic expressions, refining them over multiple generations using selection, mutation, and crossover.

## Features and Methodology

This project is designed around a tree-based representation of mathematical expressions, where nodes represent operators, variables, and constants. A genetic programming framework is used to evolve these expressions over generations, improving their ability to approximate the dataset through iterative refinements. Key features include:

- **Tree-based Representation**: Nodes represent operators, variables, and constants.
- **Genetic Programming Framework**: Evolves expressions over generations.
- **Mutation Mechanisms**: Includes subtree replacement, point mutation, hoisting, and permutation.
- **Crossover Operations**: Swaps subtrees between individuals to combine traits.
- **Fitness Evaluation**: Uses Mean Squared Error (MSE) between predicted output and actual target values.
- **Support for Mathematical Operations**: Includes elementary arithmetic functions, trigonometric functions, logarithms, exponentials, and power functions.

## Project Structure

The core of this project is built around a modular and extensible design. Key components include:

- **node.py**:

  - Defines the fundamental building blocks of symbolic expressions.
  - Implements the Node class, representing an individual element within an expression tree.
  - Supports binary operators, unary operators, variables, and constants.
  - Includes deep cloning and conversion to a human-readable string representation.

- **tree.py**:

  - Implements the Tree class, providing functionality for generating, evolving, and evaluating expression trees.
  - Includes methods for random tree generation to ensure diversity in the initial population.
  - Applies mutation and crossover operations to improve expressions over generations.
  - Evaluates fitness through MSE computation.
  - Provides visualization tools to represent evolving mathematical expressions graphically.

- **sym_reg_gp.ipynb**:
  - A Jupyter Notebook serving as the primary interface for conducting experiments.
  - Loads datasets and applies symbolic regression using the implemented genetic programming framework.
  - Presents results with visualizations.
  - Offers an interactive way to experiment with different configurations and observe the evolution of symbolic expressions over time.

## Testing and Validation

Extensive testing ensured the robustness and accuracy of the symbolic regression framework. Unit tests validated fundamental operations like mutation strategies, tree cloning, and fitness computation, ensuring trees maintained valid structures after genetic modifications.

We tested the model on different datasets with various settings to see how well it adapts. The generated expressions were compared to known functions, showing the framework’s ability to find meaningful patterns. We used MSE to measure how close the approximations were to the target values.

We also compared results from the training data with those from the testing data to check generalization. This showed that the evolved expressions stayed accurate and understandable even with new data, proving the framework's effectiveness in real-world scenarios.

## Final Results and Observations

The project successfully evolved symbolic expressions that closely matched the target functions across various test datasets. In many cases, the evolved expressions achieved low MSE, except for problem 2 and 8, indicating that the GP framework was effective in identifying complex mathematical relationships.

One of the key observations during the experimentation phase was the importance of maintaining a balance between exploration and exploitation. A well-tuned mutation strategy ensured that the search space was thoroughly explored, while selective pressure on fitness helped refine promising solutions.

## Formulas from Problem 0 to 8

### Problem 0

```markdown
np.subtract(np.reciprocal(np.reciprocal(x[0])), np.tanh(np.tanh(np.tanh(np.multiply(x[1], -0.1817006964170753)))))
```

### Problem 1

```markdown
np.sin(x[0])
```

### Problem 2

```markdown
np.multiply(np.cosh(np.negative(np.square(-2.891705823115304))), np.multiply(np.arctan(np.add(np.add(x[1], x[2]), np.add(x[0], x[0]))), np.cosh(np.square(np.cosh(-1.70765096352428)))))
```

### Problem 3

```markdown
np.add(np.multiply(np.absolute(np.maximum(x[1], -4.973875097263662)), np.multiply(np.minimum(np.negative(x[1]), x[1]), np.maximum(x[1], -4.973875097263662))), np.subtract(np.subtract(np.cosh(np.minimum(4.365965511080824, x[0])), np.add(np.sinh(-2.0174843148004316), x[2])), np.add(np.add(x[2], -1.9699415189846148), np.add(np.remainder(x[0], -2.07102849364453), np.divide(x[2], 0.6608317890232218)))))
```

### Problem 4

```markdown
np.subtract(np.maximum(np.arccos(np.tanh(np.multiply(5.5921039198210485, x[0]))), np.arccos(np.tanh(np.divide(-4.350344424685952, x[0])))), np.minimum(np.multiply(np.add(-2.5649473664638034, np.minimum(-4.786393096058477, x[0])), np.cos(x[1])), np.multiply(np.cos(x[1]), np.add(np.cos(x[1]), np.subtract(-5.648104136297606, 0.041896408583709466)))))
```

### Problem 5

```markdown
np.multiply(np.minimum(np.multiply(np.add(np.subtract(x[0], x[1]), np.maximum(2.2612182619474845, x[0])), np.divide(np.subtract(3.495395070093316, x[1]), np.reciprocal(x[0]))), np.arctan(np.divide(np.reciprocal(3.9475633927744758), np.power(x[1], x[1])))), np.square(-2.180789186394816e-05))
```

### Problem 6

```markdown
np.add(np.add(np.multiply(np.add(x[1], np.log(1.1135794785464643)), np.maximum(-0.0974477804335967, -4.8350566121696135)), np.reciprocal(np.divide(np.exp(0.09836178124853223), np.minimum(x[1], 3.1764013732538716)))), np.add(np.multiply(x[0], np.add(np.maximum(-0.7971972842251871, -4.79648789365918), np.sin(3.0327581081551704))), np.add(np.multiply(np.multiply(1.1761942159177892, x[1]), -0.0974477804335967), x[1])))
```

### Problem 7

```markdown
np.cosh(np.subtract(-2.280284342585475, np.multiply(np.maximum(np.add(-0.0416842700855411, x[0]), np.minimum(-1.2412509359118018, x[1])), np.maximum(x[1], np.minimum(x[0], -0.716333093981695)))))
```

### Problem 8

```markdown
np.multiply(np.subtract(np.add(114.57246359623406, np.exp(x[5])), np.subtract(np.sinh(x[5]), np.multiply(-4.708598611033789, -5.42828459880723))), np.subtract(np.subtract(np.minimum(np.subtract(x[4], -3.4939274899130344), np.tanh(x[3])), np.subtract(np.cbrt(x[5]), np.sinh(x[5]))), np.subtract(np.divide(np.square(x[4]), 2.579579025259571), np.multiply(x[5], 0.9293568845654132))))
```
