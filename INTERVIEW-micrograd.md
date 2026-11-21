# Interview notes — micrograd

Prep material for `micrograd_annotated.py`. Written to be read out loud,
roughly: pitch first, then the mechanisms, then the gaps — because knowing your
own edges is what distinguishes you from someone who followed a tutorial.

> The RAG project has its own notes, in the `german_RAG` repo at
> `Rag/INTERVIEW.md`.

---

## Contents

- [The project](#the-project)
- [Concepts you must be able to explain cold](#concepts-you-must-be-able-to-explain-cold)
- [Numbers to know without looking](#numbers-to-know-without-looking)
- [Weak spots — own them before they're found](#weak-spots--own-them-before-theyre-found)

---

## The project

**File:** `micrograd_annotated.py`. Pure standard library — only `math` and
`random`. No numpy, no PyTorch.

### The 30-second pitch

> A scalar-valued reverse-mode automatic differentiation engine, plus a small
> neural network library on top of it. Every arithmetic operation on a `Value`
> builds a node in a computation graph that remembers its parents and carries a
> closure implementing its own local chain rule. Calling `.backward()`
> topologically sorts the graph and walks it in reverse, accumulating
> gradients. On top of that sit `Neuron`, `Layer` and `MLP` — about 37
> parameters for a 2→4→4→1 network — and it trains with plain gradient descent.
> I wrote it to understand what PyTorch's `autograd` is actually doing.

### What's in it

| Class | Role |
|---|---|
| `Value` | A scalar that remembers the op that produced it. `+ * ** - /`, unary negation, `relu()`, `tanh()` |
| `Neuron` | Weighted sum of inputs + bias, through `tanh` |
| `Layer` | A list of neurons all seeing the same input |
| `MLP` | A chain of layers |

The demo script runs three things: a gradient sanity check against hand-derived
derivatives for `L = (a*b) + 2*(a+b)`, a forward pass, then 50 steps of gradient
descent on a 4-point toy binary classification set.

### The two mechanisms to be able to explain at a whiteboard

**1. Why reverse topological order?** The topological sort guarantees every node
appears after its children. Walking it in *reverse* means that when you call a
node's `_backward()`, every downstream use of that node has already contributed
its gradient. If you walked it in any other order you'd propagate a partial
gradient and silently get wrong answers — not a crash, just wrong numbers.

**2. Why `+=` and not `=` for gradients?** A value used in two places receives a
contribution from each path. `self.grad += ...` accumulates them;
`=` would keep only the last one. This is the multivariate chain rule: the total
derivative is the *sum* over all paths. Corollary an interviewer may push on:
this is exactly why real frameworks make you call `optimizer.zero_grad()` —
gradients accumulate by design, so you must clear them between steps.

### Likely questions

**"Why scalars? Isn't that hopelessly slow?"**
Yes, and deliberately. One `Value` per number means the graph is visible — you
can print it and read the chain rule off it. Real frameworks do exactly the same
thing with tensors as the unit instead of scalars, which turns per-element Python
closures into single vectorised kernels. The *algorithm* is identical; the
granularity is the whole performance story.

**"What does `tanh` buy you over `relu`?"**
Both are here. `tanh` is zero-centred and bounded in (-1, 1), which suits the toy
targets of ±1; it saturates, so gradients vanish for large inputs. `relu` doesn't
saturate on the positive side but can die at zero. The reason to implement both is
that each needs its own `_backward`, and writing `1 - t**2` for tanh is a good
check that you actually understand the local derivative.

**"Walk me through what `.backward()` does."**
Three steps: (1) topologically sort the graph so every node follows its children;
(2) seed the output's gradient at 1.0, since dL/dL = 1; (3) walk the list in
reverse calling each node's `_backward()` closure, which applies that op's local
derivative and accumulates into its parents.

**"Your training run doesn't converge."**
Correct, and it's intentional — loss ~1.5 after 50 steps, 3 of 4 points right.
The learning rate, step count and dataset are kept small so the loop stays
readable. Raising the step count or learning rate fits all four. I'd rather ship
a readable loop with an honest note than a tuned one that obscures the mechanism.

---

---

## Concepts you must be able to explain cold

Short, correct answers. If you can't say these without hedging, review them.

| Concept | The answer in one breath |
|---|---|
| **Automatic differentiation** | Compute exact derivatives by recording the operations as a graph and applying the chain rule mechanically — not finite differences, not symbolic algebra. |
| **Reverse mode** | One backward pass gives every input's gradient at once. Cheap when there are many inputs and one output, which is exactly the shape of a loss function. |
| **Forward mode** | One pass per *input*. Cheaper when inputs are few and outputs many — the opposite shape. |
| **Topological sort** | An ordering where every node follows the nodes it depends on. Reversing it is what guarantees a gradient is fully accumulated before it propagates. |
| **Why `+=` for gradients** | A value used in several places gets a contribution from each path; the total derivative is the *sum* over paths. This is why frameworks need `zero_grad()`. |
| **Vanishing gradient** | `tanh` saturates, so its derivative `1 - t**2` goes to zero for large inputs and updates stall. |
| **Why a tensor framework is faster** | Same algorithm, coarser unit: one kernel over a whole array instead of a Python closure per scalar. |

---

## Numbers to know without looking

**micrograd:** 37 parameters in a 2→4→4→1 MLP · 50 training steps · loss ~1.5 ·
3 of 4 points correct.

37 parameters breaks down as: layer 1 is 4 neurons × (2 weights + 1 bias) = 12,
layer 2 is 4 × (4 + 1) = 20, layer 3 is 1 × (4 + 1) = 5.

---

## Weak spots — own them before they're found

Volunteering these reads as judgement, not weakness. Being caught not knowing
them reads as the opposite.

- **No `__rsub__` / `__rtruediv__`.** `a - 2` works, `2 - a` raises TypeError.
  `__radd__` and `__rmul__` are there; the subtraction and division mirrors
  aren't. A two-line fix I haven't made.
- **No `exp` or `log`,** so no softmax and no cross-entropy — it's squared error
  and `tanh` only.
- **A comment says "mean squared error" where the code computes a *sum* of
  squared errors** (`sum((yout - ygt) ** 2 ...)`). Harmless for training —
  it just scales the effective learning rate — but the comment is wrong and
  worth fixing.
- **Scalar granularity makes it unusably slow at real scale.** Intentional;
  know the tensor answer from Project 1's Q&A.

---

## If you only remember one thing

**micrograd:** reverse topological order is what guarantees a node's gradient is
fully accumulated before it propagates further back — and `+=` rather than `=`
is the multivariate chain rule, which is why real frameworks need
`zero_grad()`.
