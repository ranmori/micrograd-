import math  # need tanh() for the activation function later


class Value:
    """A scalar value that remembers the operation that created it,
    so it can compute gradients automatically (reverse-mode autodiff)."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data                # the actual number this Value holds, e.g. 2.0
        self.grad = 0.0                 # dLoss/dThis -- starts at 0, filled in by backward()
        self._backward = lambda: None   # function that pushes this Value's grad to its children;
                                         # default does nothing (leaf nodes have no children to update)
        self._prev = set(_children)     # the Value(s) that were used to compute this one
                                         # e.g. if out = a + b, then out._prev = {a, b}
        self._op = _op                  # string like '+' or '*', just for debugging/printing

    def __add__(self, other):
        # lets us write `a + b` where a, b are Values -- Python calls this automatically
        other = other if isinstance(other, Value) else Value(other)
        # ^ so `a + 5` also works: wraps the plain number 5 into a Value first
        out = Value(self.data + other.data, (self, other), '+')
        # out.data is the actual sum; (self, other) records "these made me"

        def _backward():
            # Chain rule: if out = self + other, then
            # d(out)/d(self) = 1  and  d(out)/d(other) = 1
            # out.grad is "how much does the final loss change per unit change in out"
            # (already computed by whoever came after `out` in the graph)
            # so self's share of that sensitivity is 1 * out.grad, same for other
            self.grad += 1.0 * out.grad     # += because self might be used in MULTIPLE
            other.grad += 1.0 * out.grad    # places in the graph -- gradients from each use ADD UP
        out._backward = _backward   # attach this function to `out`, to be called later by backward()
        return out

    def __mul__(self, other):
        # lets us write `a * b`
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # if out = self * other, then
            # d(out)/d(self) = other.data   (calculus: derivative of x*y w.r.t. x is y)
            # d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        # lets us write `a ** 2` (used for squared-error loss later)
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        # ReLU(x) = x if x > 0, else 0 -- a common "does this neuron fire" nonlinearity
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # gradient is 1 where the neuron was "on" (data > 0), and 0 where it was clamped to 0
            # (out.data > 0) evaluates to True/False, which Python treats as 1/0 in arithmetic
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # squashes any number into the range (-1, 1) -- our neuron's activation function
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # derivative of tanh(x) is 1 - tanh(x)^2 -- a standard calculus identity
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # --- convenience operators, so we can write a - b, -a, a / b naturally ---
    def __neg__(self): return self * -1                # -a  ==  a * -1
    def __sub__(self, other): return self + (-other)   # a - b  ==  a + (-b)
    def __radd__(self, other): return self + other      # handles `5 + a` (plain number first)
    def __rmul__(self, other): return self * other      # handles `5 * a`
    def __truediv__(self, other): return self * other**-1   # a / b  ==  a * b^(-1)

    def backward(self):
        # This is the actual backpropagation algorithm.
        # Goal: visit every Value in the graph in the RIGHT ORDER so that by the time
        # we process a node, everything that depends on it has already pushed its gradient in.

        # Step 1: topological sort -- build a list where every child appears
        # BEFORE its parent (i.e. before anything that was computed FROM it)
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)        # recurse into children first
                topo.append(v)          # then add this node -- so children end up earlier in the list
        build(self)

        # Step 2: seed the gradient at the very end of the graph.
        # self here is the final loss -- dLoss/dLoss is trivially 1
        self.grad = 1.0

        # Step 3: walk the list BACKWARDS (from output toward inputs),
        # calling each node's _backward() to push its gradient onto its children.
        # By reversing topo, we guarantee a node's grad is fully accumulated
        # (from ALL its uses) before we call its _backward().
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        # controls what print(some_value) shows -- just for readability
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ============================================================
# Sanity check: does our hand-built gradient match calculus?
# ============================================================
a = Value(2.0)
b = Value(-3.0)
c = a + b          # c = a + b               -> dc/da = 1, dc/db = 1
d = a * b          # d = a * b               -> dd/da = b, dd/db = a
L = d + c * 2      # L = (a*b) + 2*(a+b)     -- this is what we'll differentiate

L.backward()       # computes dL/da and dL/db automatically, for EVERY Value in the graph
print("L =", L)
print("dL/da =", a.grad, " (expect b + 2 = -1.0)")   # by hand: dL/da = b + 2*1 = -3 + 2 = -1
print("dL/db =", b.grad, " (expect a + 2 =  4.0)")   # by hand: dL/db = a + 2*1 =  2 + 2 =  4


# ============================================================
# Now: stack Values into a neuron, a layer, and a full network
# ============================================================
import random
random.seed(42)   # fixed seed so results are reproducible every run

class Neuron:
    """One neuron: out = tanh(w1*x1 + w2*x2 + ... + b)"""
    def __init__(self, n_inputs):
        # one random weight per input, plus one bias -- all as Value objects
        # so every one of them can receive a gradient
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        # this is what runs when you do `neuron(some_input)`
        # weighted sum: w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # ^ zip pairs up each weight with its matching input; sum starts from self.b (the bias)
        return act.tanh()   # squash through the nonlinearity

    def parameters(self):
        # every trainable number this neuron owns -- needed so the training loop
        # knows what to update
        return self.w + [self.b]

class Layer:
    """A layer is just a list of neurons, each seeing the same inputs."""
    def __init__(self, n_inputs, n_outputs):
        # n_outputs neurons, each independently wired to all n_inputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]     # run every neuron on the same input x
        return outs[0] if len(outs) == 1 else outs   # unwrap single-neuron layers for convenience

    def parameters(self):
        # flatten: every parameter of every neuron in this layer
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """A multi-layer perceptron: chain of Layers."""
    def __init__(self, n_inputs, layer_sizes):
        # e.g. MLP(2, [4, 4, 1]) means: 2 inputs -> layer of 4 -> layer of 4 -> layer of 1
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        # feed x through each layer in sequence -- output of one is input to the next
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # every trainable weight and bias in the whole network
        return [p for layer in self.layers for p in layer.parameters()]


# --- quick check: build a 2-input, [4, 4, 1] network and run one forward pass ---
net = MLP(2, [4, 4, 1])
x = [Value(0.5), Value(-0.3)]
out = net(x)
print("\nnetwork output for x=[0.5, -0.3]:", out)
print("total trainable parameters:", len(net.parameters()))


# ============================================================
# Train on a toy binary classification dataset
# 4 points in 2D, labels are +1 or -1
# ============================================================
xs = [
    [2.0, 3.0],
    [3.0, -1.0],
    [0.5, 1.0],
    [1.0, 1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]   # desired output for each row of xs, same order

net = MLP(2, [4, 4, 1])       # fresh network, untrained

print("\n--- training ---")
for step in range(50):
    # 1. forward pass: run every input through the (current) network
    ypred = [net(x) for x in xs]

    # 2. loss: mean squared error -- how far off are our predictions, squared and summed
    #    squaring makes big errors count more, and makes the loss always positive
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # 3. zero out old gradients before this step's backward pass.
    #    grad ACCUMULATES (+=) inside our Value class, so without this reset,
    #    gradients from step 1 would still be sitting there polluting step 2
    for p in net.parameters():
        p.grad = 0.0

    # 4. backward pass: this single call computes dLoss/dParam for
    #    EVERY parameter in the entire network, via the chain rule
    loss.backward()

    # 5. gradient descent: nudge every parameter a little bit AGAINST its gradient
    #    (grad points toward increasing loss, so we subtract to decrease loss)
    learning_rate = 0.05
    for p in net.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  loss = {loss.data:.4f}")

print("\nfinal predictions (target: 1, -1, -1, 1):")
for x, ygt in zip(xs, ys):
    print(f"  input {x} -> predicted {net(x).data:.3f}  (target {ygt})")
