# micrograd

A small scalar-valued automatic differentiation engine, built from scratch in pure Python, plus a tiny neural network library on top of it.

Every line is annotated with the reasoning behind it — this is a learning implementation, written to be read.

## What's in here

`micrograd_annotated.py` contains everything:

- **`Value`** — a scalar that remembers the operation that produced it, forming a computation graph. Supports `+`, `*`, `**`, `-`, `/`, unary negation, plus `relu()` and `tanh()` activations. Calling `.backward()` on any `Value` runs reverse-mode autodiff over the whole graph via a topological sort.
- **`Neuron`** — a weighted sum of inputs plus a bias, squashed through `tanh`.
- **`Layer`** — a list of neurons that all see the same input.
- **`MLP`** — a chain of layers.

No numpy, no PyTorch. Only the standard library (`math`, `random`).

## Running it

```bash
python micrograd_annotated.py
```

The script runs three things in order:

1. **A gradient sanity check.** It builds `L = (a*b) + 2*(a+b)` and confirms the computed gradients match the ones you get by hand.
2. **A forward pass** through a freshly initialized 2 → 4 → 4 → 1 network (37 trainable parameters).
3. **A training loop** — 50 steps of gradient descent on a 4-point toy binary classification dataset.

## How backprop works here

Each operation on a `Value` builds a new `Value` that records its children and attaches a `_backward` closure implementing the local chain rule for that op. `backward()` then:

1. Topologically sorts the graph so every node appears after its children.
2. Seeds the output gradient at `1.0` (since `dL/dL = 1`).
3. Walks the sorted list in reverse, calling each `_backward()`.

Reversing the topological order is what guarantees a node's gradient is fully accumulated from *all* its downstream uses before it gets propagated further back. Gradients use `+=` rather than `=` for the same reason — a value used in multiple places receives a contribution from each.

## A note on the training results

The toy run reaches a loss around 1.5 after 50 steps and gets 3 of the 4 points right. It isn't tuned to convergence — the learning rate, step count, and dataset are deliberately small so the loop stays readable. Raising the step count or the learning rate will fit all four points.

## Credit

The design follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). This is a from-scratch reimplementation written as a learning exercise, with commentary added throughout.
