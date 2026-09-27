# karpathy_rnn_lab

An executable, test-driven companion to Andrej Karpathy's
[*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
(2015) — read the post section by section, run the matching test file
alongside it, and see the claim demonstrated instead of taking it on faith.
No notebook: the tests *are* the exploration surface.

It also implements the "minimal character-level RNN language model in
Python/numpy" the post links to (the 100-line gist), rebuilt as a small
domain-driven, functional-core / imperative-shell design.

## Why this structure

- **Domain (`domain/`)** — pure functions and immutable value objects only:
  `Vocabulary`, `RNNParams`, `Gradients`, `AdagradMemory`. Every function
  takes a state and returns a *new* state; nothing here mutates an argument
  or does I/O. This is the "functional core."
- **Application (`application/train_service.py`)** — the one place allowed
  to loop and accept a side-effecting callback (`on_snapshot`). It threads
  an immutable `TrainerState` through the domain functions. This is the
  "imperative shell."
- **Infrastructure (`infrastructure/corpus.py`)** — where training text
  comes from (a file, or a small built-in default). The domain layer never
  knows or cares.
- **`quality/crap.py`** — composes `radon` (cyclomatic complexity) and
  `coverage.py` (line coverage) into the published CRAP formula, so
  "quality < 5" is something you can actually run, not just assert.

The domain layer's internal loops (`bptt`, the Adagrad `_adagrad_step`
closure) do accumulate into local numpy arrays for performance — the same
way a hand-written BPTT loop would — but that's invisible to callers, who
only ever get back a fresh `Gradients` / `RNNParams` / `AdagradMemory`.
Purity is a property of the function's boundary, not a ban on local
bookkeeping.

## Running it

```bash
pip install -r requirements.txt
pytest                       # run everything (~1-2s)
pytest tests/test_forward_pass.py -v   # run just one section's tests

coverage run -m pytest       # for the CRAP score
coverage json -o coverage.json
python quality/crap.py
```

> `radon`/`coverage` could not be installed or executed in the sandbox that
> produced this project (no network access there), so `quality/crap.py`
> is unverified end-to-end — everything else (all 30 tests, via a plain
> assert-based shim standing in for pytest) was run and passes. If
> `quality/crap.py` errors on your machine, run `radon cc -j domain
> application` once and check its JSON keys match what `_run_radon()`
> expects (`name`, `lineno`, `endline`, `complexity`); that's the one part
> that depends on radon's exact version.

## Test file ↔ article section map

| Test file | Article section | What it demonstrates |
|---|---|---|
| `test_vocabulary.py` | *Character-Level Language Models* | 1-of-k ("one hot") encoding of a small vocabulary |
| `test_forward_pass.py` | *RNN computation* | `h = tanh(Whh·h + Wxh·x + bh)`; output is a real probability distribution; **the hidden state actually matters** — same input, different `h`, different output |
| `test_loss.py` | *"a more technical explanation ... Softmax classifier"* | cross-entropy loss behaves as expected: max at uniform, low when confident-and-correct, high when confident-and-wrong |
| `test_gradient_check.py` | *"we can run the backpropagation algorithm ... to figure out in what direction we should adjust every weight"* | a numerical gradient check — the same technique Karpathy's own reference gist ships — proving BPTT's analytic gradients are wired correctly |
| `test_gradient_clipping.py` | (reference implementation, not named in the prose) | gradients get bounded so one bad batch can't blow up training |
| `test_adagrad_update.py` | *"per-parameter adaptive learning rate methods"* | Adagrad moves parameters against the gradient, and a parameter that has already seen big gradients gets a smaller effective step |
| `test_sampling.py` | *"At test time ... we sample ... and feed it right back in"* + *Temperature* | sampling is reproducible under a fixed RNG seed; low temperature sharpens toward the favorite character, high temperature flattens the distribution |
| `test_training_evolution_e2e.py` | *The evolution of samples while training* + the post's central claim | loss on a fixed window falls by >5x after training; a low-temperature sample from the trained model reproduces whole words it was never told about explicitly — only shown one character at a time |

## A note on the training corpus

`infrastructure/corpus.py` ships a short, repetitive, originally-written
string rather than a scraped/copyrighted text (Karpathy trained on Paul
Graham essays, Shakespeare, Wikipedia, Linux source...) — that keeps the
end-to-end test fast and license-clean. Point `load_corpus("/path/to/text")`
at any text file of your own to reproduce something closer to the post's
actual experiments; it'll just need far more than 8 training epochs to get
there.
