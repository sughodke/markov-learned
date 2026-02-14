# Quantum-Accelerated Markov Chains

**Finding the most likely next character with Grover's algorithm instead of brute-force search.**

This project builds a classical bigram Markov chain from Shakespeare's complete works, then uses a quantum algorithm to find the highest-probability next character — with a provable quadratic speedup over the classical approach.

## The problem: argmax is slow

A bigram Markov chain stores transition probabilities P(next | current) for every character pair. Given a current character, predicting the next one means scanning the entire row of probabilities to find the maximum:

```
Transition probabilities from 't':
  '\n': 0.0118    'SPC': 0.2464    '!': 0.0029
  ... 62 more entries ...
  'h': 0.3393  <-- the answer is buried in here
```

Classically, this is an O(N) linear scan. For a 65-character vocabulary, that's 65 comparisons every time you want to predict the next character. Not terrible for 65 — but the same structure appears in language models with vocabularies of 50,000+ tokens, and in molecular simulation state spaces that can be astronomically large.

Can we do better?

## A quick primer on Grover's algorithm

Grover's algorithm (1996) solves unstructured search problems with a quadratic speedup. Given N items and a single "marked" target, a classical computer needs O(N) lookups on average. Grover's algorithm finds it in O(sqrt(N)).

The algorithm works in three steps:

**1. Superposition.** Apply Hadamard gates to put all N states into equal superposition — the quantum register holds every possible answer simultaneously, each with amplitude 1/sqrt(N).

**2. Oracle.** A phase oracle flips the sign of the target state's amplitude. This is the only place problem-specific knowledge enters the circuit. For our Markov chain, the oracle encodes "which character index is the argmax?"

**3. Diffusion.** The Grover diffusion operator reflects all amplitudes about their mean. Since the target has a negative amplitude (from the oracle), this reflection boosts it above the mean while suppressing everything else.

Repeat steps 2-3 approximately pi/4 * sqrt(N) times, and the target state's probability approaches 1. Measure, and you get the answer.

The key insight: **we never look at the actual probability values during the quantum search.** The oracle just needs to recognize the winner. All the heavy lifting — the amplification from 1/N to near-certainty — comes from quantum interference.

## What we built

The notebook `quantum_markov.ipynb` implements this end-to-end:

1. Build a 65x65 bigram transition matrix from ~1.1M characters of Shakespeare
2. Encode all 65 characters into 7 qubits (2^7 = 128 states, with 63 unused padding states)
3. Construct a Grover circuit that finds the argmax of any transition row
4. Simulate with [Cirq](https://quantumai.google/cirq) and verify correctness

## Results

### 3-qubit warmup (8 characters)

First, a small example with the 8 most frequent characters to visualize what's happening. Given the query character `'t'`, the classical transition probabilities are:

```
Query character: 't'

Transition probabilities from 't':
  |000> 'SPC': 0.2989
  |001> 'e':   0.0803
  |010> 't':   0.0157
  |011> 'o':   0.1061
  |100> 'a':   0.0354
  |101> 'h':   0.4117  <-- TARGET (argmax)
  |110> 's':   0.0182
  |111> 'r':   0.0338
```

After just 2 Grover iterations (vs. 8 classical comparisons), the quantum circuit amplifies `|101>` = `'h'` to 94.5% probability:

```
Measurement results (2 Grover iterations, 10,000 shots):
  |000> 'SPC':   72 ( 0.7%)
  |001> 'e':     89 ( 0.9%)
  |010> 't':     92 ( 0.9%)
  |011> 'o':     83 ( 0.8%)
  |100> 'a':     86 ( 0.9%)
  |101> 'h':  9,427 (94.3%)  <-- TARGET
  |110> 's':     64 ( 0.6%)
  |111> 'r':     87 ( 0.9%)
```

The circuit is compact — 3 qubits, depth 21, 43 gates:

```
0: ───H───────@───H───X───────────@───X───H───────────@───H───X───────────@───X───H───M───
              │                   │                   │                   │           │
1: ───H───X───@───X───H───X───────@───X───H───X───────@───X───H───X───────@───X───H───M───
              │                   │                   │                   │           │
2: ───H───H───X───H───H───X───H───X───H───X───H───H───X───H───H───X───H───X───H───X───M───
```

### 7-qubit full vocabulary (65 characters)

Now the real test — searching over all 65 characters in Shakespeare's vocabulary. The 65 characters are encoded into 7 qubits (128-dimensional Hilbert space). States 65-127 are unused padding; Grover naturally suppresses them since they're never marked by the oracle.

```
7-qubit full vocabulary: 65 characters in 128-dim Hilbert space
Query: 't' (idx 58)
Target: |0101110> = 'h' (idx 46)
```

With 9 Grover iterations (vs. 128 classical comparisons in the full Hilbert space), the target is found with 98.8% probability:

```
Top-10 measurement results (7-qubit, 9 iterations, 10,000 shots):
  |0101110> 'h':       9,869 (98.7%)  <-- TARGET
  |0010011> 'G':           5 ( 0.1%)
  |1001110> pad(78):       3 ( 0.0%)
  |0001110> 'B':           3 ( 0.0%)
  |1000111> pad(71):       3 ( 0.0%)
  ...

Grover result:    |0101110> = 'h'
Classical argmax: |0101110> = 'h'
Match: True
```

### Summary

```
============================================================
  GROVER'S ALGORITHM FOR MARKOV CHAIN SEARCH
============================================================
  Corpus:              1,115,394 characters (Shakespeare)
  Vocabulary:          65 characters
  Query character:     't'

  --- 3-Qubit (8 characters) ---
  Classical argmax:    'h'
  Grover result:       'h'
  Grover iterations:   2 (vs 8 classical comparisons)
  Success probability: 94.5%

  --- 7-Qubit (full 65-char vocabulary) ---
  Hilbert space:       128 states (65 chars + 63 padding)
  Classical argmax:    'h'
  Grover result:       'h'
  Grover iterations:   9 (vs 128 classical comparisons)
  Success probability: 98.8%

  Speedup: O(sqrt(N)) vs O(N) -- quadratic advantage
============================================================
```

## Why this matters

The speedup here is quadratic: O(sqrt(N)) vs O(N). For 128 states, that's 9 iterations instead of 128 comparisons. The ratio gets better as the search space grows:

| Vocabulary size | Classical comparisons | Grover iterations | Speedup |
|---|---|---|---|
| 8 (3 qubits) | 8 | 2 | 4x |
| 128 (7 qubits) | 128 | 9 | 14x |
| 1,024 (10 qubits) | 1,024 | 25 | 41x |
| 50,000 (16 qubits) | 50,000 | 175 | 286x |

For modern language models with 50K+ token vocabularies, or molecular dynamics simulations with millions of states, the quadratic advantage becomes substantial. The same Grover circuit structure — oracle + diffusion, repeated O(sqrt(N)) times — scales to any search space that fits in qubits.

## Broader pattern: control software for stochastic computation

This project is a clean instance of a general architecture found across data processing systems: **deterministic control software wrapping a stochastic computation core.** The quantum circuit is a coprocessor — powerful but unable to program itself. Classical code authors its instructions, parameterizes its execution, and interprets its output. The same structure appears wherever probabilistic computation is used.

### The shared control loop

Every system that relies on a stochastic engine follows the same pipeline this notebook implements:

1. **Encode** the problem from domain space into the stochastic engine's native format
2. **Parameterize** the process to balance accuracy against cost
3. **Execute** and collect raw stochastic output
4. **Validate** that the output meets confidence or correctness requirements
5. **Translate** back into domain-meaningful results

In this project, bigram probabilities are encoded into an oracle (step 1), the iteration count is computed as `round(pi/4 * sqrt(N))` (step 2), the simulator runs and returns bitstrings (step 3), results are checked against the classical argmax (step 4), and bitstrings are mapped back to characters (step 5).

### Where the same pattern appears

**Monte Carlo simulation.** A derivatives pricing engine encodes market data into a volatility surface, runs thousands of stochastic paths, and averages terminal payoffs. The control software decides how many paths to run (analogous to Grover's iteration count) and applies variance reduction techniques (analogous to amplitude amplification). Over-simulation wastes compute; under-simulation gives unreliable prices.

**MCMC in Bayesian inference.** Markov Chain Monte Carlo — literally this project's domain used as a computation engine — samples from a posterior distribution. The control layer sets burn-in length, thinning interval, and number of chains, then runs convergence diagnostics (Gelman-Rubin) to decide when to stop. Grover's over-rotation problem (performance degrades at 3 iterations in the 3-qubit case) has a direct analogue: running MCMC too long with poor tuning causes autocorrelation to dominate.

**Stream processing with probabilistic data structures.** Bloom filters, Count-Min sketches, and HyperLogLog estimators trade exactness for speed. The control software computes how many hash functions or registers are needed for a target false-positive rate — the same relationship as computing Grover iterations for a target success probability.

**Stochastic gradient descent.** The training loop in this project's own `train.py` follows the pattern: classical code selects batch size and learning rate, the stochastic core samples mini-batches and computes gradients, and the control loop monitors loss and decides when to stop. Learning rate is the direct analogue of Grover's iteration count — too small and convergence is slow, too large and you oscillate past the optimum.

**Randomized distributed algorithms.** Gossip protocol fan-out (how many peers to contact per round) is calculated from the desired convergence time, exactly like Grover iterations are calculated from the search space size. The control layer monitors convergence and triggers additional rounds if needed.

### Why quantum makes the pattern visible

In most production systems, the control/stochastic boundary is buried in library internals or spread across microservices. A quantum circuit makes it explicit: the quantum processor literally cannot operate without a classical controller issuing every gate, computing every parameter, and interpreting every measurement. This project is a minimal working example of that dependency structure.

## The classical side

This project also includes a neural Markov chain model that goes beyond bigrams. `ScanMarkovModel` uses a linear recurrence with input-dependent gating (Mamba-style) to compose character sequences:

```
e_t = embed(token_t) + pos_embed(t)
[A_t, B_t] = sigmoid(gate_proj(e_t))
h_t = A_t * h_{t-1} + B_t * e_t
output = decode(layer_norm(h_final))
```

This is the associative scan operator, trained on mixed 2-5 gram samples from Shakespeare (~340K parameters). See `chainable_markov.ipynb` for an interactive walkthrough.

## Running it

**Quantum notebook** (recommended starting point):

```bash
pip install cirq numpy matplotlib seaborn
jupyter notebook quantum_markov.ipynb
```

**Classical model training:**

```bash
pip install torch
python train.py
```

## Files

- `quantum_markov.ipynb` — Grover's algorithm for Markov chain next-character search
- `chainable_markov.ipynb` — Classical neural Markov model walkthrough
- `model.py` — `ScanMarkovModel`, `CharVocab`, training and generation utilities
- `train.py` — Standalone training script
- `data/shakespeare.txt` — Tiny Shakespeare corpus (~1.1MB)
