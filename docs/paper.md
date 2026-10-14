---
title: "ep4nn: evidence-gated structural growth in a connectome-compiled equilibrium network"
author: "Hefei Jiuzhai Big Data Technology / Lituo Cloud Intelligence Technology (Nanjing)"
date: 2026-10-14
---

# Abstract

We present **ep4nn**, a continuing network that begins as the measured 302-neuron
*C. elegans* wiring imported into a rate-based equilibrium model and, when it meets two
situations it cannot tell apart, appends neurons and synapses under three gates: a
**prerequisite gate** (does the answer move at all), an **evidence gate** (one collision is
noise), and an **outcome gate** (growth that does not reduce the collision is rolled back).
The system is built on the released Cadence library and reuses two practices from the
open fly-connectome projects: protocols written before their data, and receipts that
separate nominal from executed scale. We report measurements rather than claims. On one
laptop CPU the library settles a 10,000-neuron, 103,943-synapse graph in 6.5 ms and a
100,000-neuron, 1,239,921-synapse graph in 77.4 ms (32 sweeps, CSR transport, about 1.76 ns
per synapse). A **state-dependent contraction certificate** exists on the worm graph where
the library's global worst-case bound fails (local spectral radius 0.529 against a worst-case
row bound of 88.0). The growth machinery behaves as specified: over 1,200 ticks it refused
growth 1,200 times because the prerequisite failed, and an earlier variant without the
prerequisite gate produced 30 growths and 30 rollbacks. The paper's main negative finding is
that prerequisite: across 60 operating points of the worm graph the declared readout responds
along a single direction (best spread 0.0777 with collision cosine 1.0), so capacity growth is
not the right repair and ep4nn declines to grow. We conclude that structural growth studies
must publish the prerequisite measurement alongside the growth result, and we propose a receipt
format that makes claims comparable across connectome projects.

# 1 Introduction

A measured connectome is a graph of who contacts whom. It is not an operating point, not a
dynamical model, and not a behaviour. Three years of open fly-connectome work demonstrate the
gap concretely: a complete male central nervous system of about 166,000 neurons is public under
CC-BY, yet the projects that run it either supply the controller, supply the semantic mapping
from cell groups to behaviour, or supply the reward. The most disciplined attempt to remove the
controller reached a negative result: driving the ventral nerve cord through two descending
neurons produced a 45.5 Hz population oscillation that a scrambled wiring reproduced as well.

Our starting question is narrower than behaviour. Suppose a network starts small, on a measured
wiring, and must **add structure** as the situations it faces accumulate. What should trigger
growth, what should growth wire, and how do we know the growth was worth it?

We answer with a system and a set of measurements. The system, `ep4nn`, treats growth as a
repair of one specific witnessable failure: two situations that the world declares different
share the same present input and produce answers that are nearly parallel at the declared
readout. The repair is a low-dimensional generator that appends neurons wired from the cells
that carry the difference, and a verdict that measures the same collisions again and rolls the
growth back when they do not improve.

The main result of this paper is that the repair is **blocked by a prerequisite we did not
expect to be the binding constraint**: on the worm graph, at 60 operating points, the declared
readout responds along a single direction. Answers differ in magnitude, not in pattern, so
every stage of growth is measuring a code that carries nothing a lesson can attach to. This is
the same failure Cadence reports for the fly antennal lobe under one global gain, and it is
detectable before any growth is attempted. We therefore state the prerequisite as a rule and
report its measurement as the paper's headline, together with the growth machinery that
enforces it.

Contributions:

1. A growth contract with three gates (prerequisite, evidence, outcome) that makes a refusal
   to grow an explicit, recorded outcome rather than a silent failure.
2. A low-dimensional structural generator whose parameters are genes, whose partner choice is
   driven by the difference between the colliding answers, and whose new neurons must reach the
   readout.
3. A state-dependent contraction certificate that certifies a settling budget on a measured
   connectome where the library's global bound does not apply.
4. Measurements of the underlying library on the worm graph and on sparse graphs up to 100,000
   neurons, including a negative control result for continuous tracking.
5. A receipt format for connectome runs, adopted from the fly projects' practice of separating
   nominal from executed scale.

# 2 Related work

**Equilibrium learning.** Cadence implements settling trained by the contrast of a free and a
nudged fixed point, the two-phase form of equilibrium propagation [Scellier and Bengio 2017] and
contrastive Hebbian learning [Movellan 1990]. The library's own documentation states the
boundary we rely on: reciprocal learning ties paired weights, and arbitrary directed imports need
not satisfy that gradient contract. Equilibrium propagation scales to deep convolutional
networks by reducing the bias of its gradient estimator [Laborieux et al. 2021], sits beside
implicit differentiation and bilevel formulations [Zucchet and Sacramento 2022], and shares its
object with deep equilibrium models [Bai et al. 2019]. Predictive-coding rules approximate
backpropagation along arbitrary computation graphs [Millidge et al. 2022; Song et al. 2020].

**Complementary memories and replay.** Fast episodic storage with slow consolidation is the
complementary learning systems proposal [McClelland et al. 1995]. Cadence implements one version:
records written once by day, fixed dreams at night taught to slow weights, and the residual
written back at dawn. Its own measurements report held-out sign accuracy rising from 0.55 to
about 0.76-0.80 on a Connect Four slice, at a night cost of roughly 140-210 s per 1,000 rows for
a 4,096-cell store, with the limit that dreams reproduce the store's errors as faithfully as its
regularities.

**Fly connectomes.** FlyWire published the adult female brain reconstruction of 139,255 neurons
[Dorkenwald et al. 2024]; the male central nervous system (brain, optic lobes, ventral nerve
cord; about 166,000 neurons) followed under CC-BY. The BANC release used by Cadence's fly lane
carries 150,802 proofread neurons and 1,877,099 connection classes. Because the ventral nerve
cord contains the leg and wing motor neurons, the male data made connectome-driven locomotion
addressable in a way the brain-only data was not. Open projects in this space differ mainly in
what they supply: a hand-wired 850-neuron stand-in with fly cell-type names for a browser demo,
a learned consequence model with an application controller, a directly read motor output whose
own receipts report that real-time neural control was not demonstrated, and an on-device
implementation that reports 10 Hz on a phone using CSR storage and per-group gating. Cadence's
own fly guide reports that at one global gain the antennal lobe ignites, that Kenyon cell codes
for two odours reach a cosine of 0.98, and that a gain per cell class brings the code to 4 and 11
percent sparsity with a cosine of 0.32.

**Conditional computation.** Mixture-of-experts routing, early exit and pruning vary the work per
input at the weight level. Dynamic structure methods vary the graph. The distinction we test is
the trigger: ep4nn triggers on a witnessable indistinguishability at the declared readout, not on
a loss, a router score, or an offline search objective.

# 3 System

## 3.1 The four rules

**Rule 1, demand.** Two situations that the world declares different, share the present input,
and produce answers at the declared readout whose absolute cosine exceeds a threshold, constitute
a capacity signal. The readout is a declared set of neurons, not the whole state; the whole state
almost always separates because it contains the driven cells themselves.

**Rule 2, evidence.** A single collision is noise. `min_evidence` collisions must be observed,
and `min_ticks_between` ticks must pass, before any structural change is attempted.

**Rule 3, growth.** A generator with a handful of genes appends `k` neurons. Each new neuron
draws in-edges and out-edges from a degree prior, prefers partners among the cells that carry the
difference between the colliding answers and their one-hop neighbourhood, keeps the graph's
excitatory fraction, adds the reverse edge with probability `reciprocal` so the equilibrium
contrast contract is preserved, and projects into the readout with probability `to_readout`.
The generator is low-dimensional by construction: the number of free parameters is far smaller
than the number of edges it can create, which is what distinguishes structural plasticity from
implicit weight learning.

**Rule 4, verdict.** After growth, the same collisions are measured again. Growth is kept only if
the collision cosine falls by `improve_margin`. Otherwise the connectome and the efficacy array
are restored from the snapshot taken before the attempt, and the attempt is counted as a failure.

## 3.2 The prerequisite gate

Before any of the above applies, the answer must move. ep4nn measures
`readout_spread`, the mean over readout cells of the maximum minus minimum activation across a
declared set of situations. Below `min_response` the system records `no_response` and does not
grow. The justification is structural: if the declared answer is constant across situations, no
amount of added capacity can reduce a collision, so growth would be the wrong repair and its
recorded outcome would carry no information.

## 3.3 The operating point

A connectome carries contact counts and transmitter signs. It does not carry excitability, and the
raw counts of the worm graph give row sums in the hundreds at which the rate model does not
settle. ep4nn declares an operating point by row-normalising the signs to `row_sum`. The library
offers two further instruments for the same purpose, `calibrate_bias` and a gain selected by
protocol, and the paper reports measurements for all three.

## 3.4 A settling budget from a state-dependent certificate

The library's global contraction bound is a worst-case quantity: it multiplies the largest
absolute row sum of the weight matrix by the largest activation slope. On a measured connectome
most neurons are inactive at the settled state, so the bound is far larger than the actual
contraction. ep4nn measures the certificate the library defines but with the slopes of the
current state,
$K = (1-\mathrm{d}t)I + \mathrm{d}t\,W\,\mathrm{diag}(s'(v))$,
and reports the spectral radius and a positive weighted witness $h$ with $Kh \le \alpha h$. The
settling budget follows as $k \approx \log(\epsilon/\delta)/\log(1/\alpha)$.

# 4 Implementation

**Data.** The worm wiring is the openworm c302 `A_Full` release (302 hermaphrodite neurons,
2,279 chemical synapse classes, 1,084 gap junctions; the gap junctions are imported in both
directions). After dropping autapses and merging parallel edges the graph has 3,137 directed
synapse classes, 10,133 contacts and a density of 3.44 percent, with 61 sensory, 91
interneuron, 118 motor, 10 command and 22 pharyngeal cells. Soma positions in the source file are
used for nothing in this paper.

**Library reuse.** The network is a `cadence.Connectome` with `cadence.Brain` on it, using the
library's rate neuron model, its free and nudged phases, its residual qualification, and its
CSR transport. No new learning rule is introduced. The growth layer sits outside the library
because `Connectome` is immutable and because the library forbids growth by design; a growth
event rebuilds the connectome, which costs one pass over the edge arrays, and carries the
learned efficacy across by regenerating the key order the library sorts by.

**Reuse from the fly projects.** Two practices are adopted. First, protocols and thresholds are
written before the run and the run happens once, following the discipline of the offline
fly-body study. Second, every run emits a receipt that separates nominal from executed scale,
following the observation that project descriptions and executed graphs differ widely.

**Receipt.** `ep4nn/receipt.py` writes a JSON receipt with six fields: dataset and hashes;
nominal and executed counts; the dynamical model and its declared parameters; whether the loop is
closed and which parts are supplied; the prerequisite measurements; and the cost.

# 5 Measurements

All measurements are from one Apple Silicon laptop, CPU only, Python 3.13, NumPy 2.5.4 and SciPy
1.18.1, single-threaded, with the released library at version 0.80.0 plus local commits
(`0b0889e`). Probes are in `probes/`, one file per table.

## 5.1 Settlement cost and scaling

| graph | neurons | synapses | mean in-degree | transport | per sweep | sweeps to 1e-4 | settle | CSR |
|---|---|---|---|---|---|---|---|---|
| worm, measured | 302 | 3,137 | 10.4 | blocked dense | 27.2 µs | 160 | 4.89 ms | n/a |
| sparse random | 1,000 | 10,337 | 10.3 | blocked dense | 161.4 µs | 32 | 5.29 ms | 0.1 MB |
| sparse random | 10,000 | 103,943 | 10.4 | CSR | 193.0 µs | 32 | 6.48 ms | 1.3 MB |
| sparse random | 100,000 | 1,239,921 | 12.4 | CSR | 2,185.8 µs | 32 | 77.4 ms | 15.3 MB |

Cost is about 1.76 ns per synapse, so the control frequency of a settling brain is bounded by
$f \approx 1/(n_{\text{syn}} \times 1.8\,\text{ns} \times \text{sweeps})$. The worm graph needs
five times as many sweeps as a random graph of the same size and degree at the same tolerance,
which locates the difficulty in the spectrum of the graph rather than in its size. Two library
defaults matter at this scale: the default constructors wire at a density of about 30 percent
independent of size (measured 29.1, 30.8, 31.7 and 32.3 percent at 152, 280, 536 and 1,048
neurons), so 10,000 neurons would carry about 30 million synapses and settle in about 1.7 s; and
the transport path is chosen by a size test rather than by density, so a 3.4 percent dense graph
pays a block array 29 times larger than its edge count. On that graph the CSR path is 1.78 times
faster end to end (987 µs against 554 µs per settle), while on a 32 percent dense graph the
blocked path is faster, which is why the choice must follow density.

## 5.2 A certificate where the global bound fails

| quantity | value |
|---|---|
| global worst-case bound $q$ | 88.0 (292 of 302 rows at or above 1) |
| local bound, current state | 1.186 (1 of 302 rows) |
| spectral radius of the local $K$ | 0.529 |
| weighted witness $\alpha$ with $Kh \le \alpha h$ | 0.696, $h \in [2.89, 11.08]$ |
| active neurons at the settled state | 294 of 302 |

The global bound gives no certificate at this gain; the state-dependent one does, and it converts
a refusal into a bounded settling budget. The same quantity is the precondition for treating the
free and nudged contrast as a gradient, which is why it is measured at run time rather than
assumed.

## 5.3 Behaviour of the growth machinery

Two variants were run for 1,200 ticks while the world increased the number of context levels that
must be distinguished.

| variant | kept | rolled back | refused, no response | neurons | self-check |
|---|---|---|---|---|---|
| without prerequisite gate | 0 | 30 | n/a | 302 | passed |
| with prerequisite gate | 0 | 0 | 1,200 | 302 | passed |

In both variants the brain never shrank, every synapse kept an efficacy, the readout stayed inside
the current graph, and every kept growth would have had to reduce its collisions. In the first
variant the collision cosine read 1.000 before and after every attempt, so the verdict layer
rolled all 30 attempts back. In the second, the prerequisite measured 0.019 and the system
declined to act, recording why.

## 5.4 The prerequisite scan, a negative result

Sixty operating points were scanned: five row sums, a gain on one cell class (none, interneurons,
command neurons), and four gain scales, over nine situations (three single senses and six
combinations of an odour with a pain level).

| rank | row sum | gain class | scale | spread | collision cosine | sweeps |
|---|---|---|---|---|---|---|
| 1 | 0.5 | command | 10 | 0.0777 | 1.000 | 224 |
| 2 | 1.0 | command | 2 | 0.0559 | 0.9999 | 64 |
| 3 | 1.0 | none | 1 | 0.0328 | 0.9998 | 64 |
| last | 2.0 | interneurons | 10 | 0.0000 | 1.000 | 96 |

The best configuration moves the readout by 0.078 across situations while leaving every pair of
context answers at a cosine of 1.0: the differences that exist lie along one direction, so the
answer has rank one with respect to the situations that must be distinguished. Several
configurations collapse to an answer that is identical for all nine situations to three decimals,
which indicates a fixed point dominated by the recurrent dynamics rather than by the input. We
report this as a negative result about the pair (measured worm wiring, rate model at a declared
operating point): **capacity growth is not the binding constraint, and a growth study on this
pair would measure an uninformative object.** The remedy is left as future work and has three
candidates, in ascending cost: a different declared readout, a dynamics with heterogeneous cells
and per-type parameters, or a gain selected by protocol against held-out physiological facts as
Cadence does for the fly.

## 5.5 A negative control result for continuous tracking

One composed brain (140 neurons, 6,028 synapses) was placed in a closed loop on a second-order
plant with five force levels, one scalar reward per tick, and 6,000 ticks per arm.

| arm | mean absolute error | last quarter | in-band fraction, last quarter | dominant action, last quarter |
|---|---|---|---|---|
| uniform random | 1.020 | 1.033 | 0.000 | uniform |
| frozen greedy | 1.011 | 1.067 | 0.467 | 1.00 |
| learning, sparse reward | 0.880 | 0.855 | 0.449 | 0.824 |
| learning, dense reward, seed 5 | 0.923 | 0.919 | 0.209 | 0.705 |
| learning, dense reward, seed 11 | 0.856 | 0.815 | 0.459 | 0.843 |
| learning every outcome | 1.016 | 1.007 | 0.000 | 0.980 |

The learned arms reduce mean error by about 13 percent and never beat the frozen baseline on the
in-band metric, and the action distribution collapses to a single level. The structural reason is
the mismatch between the credit window and the plant: the composed brain's eligibility span is 12
steps, while the closed-loop time constant of the plant is 50 to 100 steps. This is consistent
with the library's own control-oriented demo, which learns a consequence model and leaves the
controller supplied.

# 6 Discussion

**What is shown.** The growth contract is implementable on a measured connectome at negligible
rebuild cost, its gates fire in the intended order, and the prerequisite measurement is what
decides whether structural work is meaningful. Two library-level results are independent of
ep4nn: a state-dependent certificate exists where the global bound fails, and a sparse graph is
settled at about 1.8 ns per synapse with a per-settle cost that makes 10,000 neurons a
real-time-scale object rather than a research curiosity.

**What is not shown.** No growth has yet been kept. The paper therefore makes no claim about the
benefit of structural growth, only about the contract that would measure it and the prerequisite
that currently blocks it. We also make no claim that the worm graph is an inappropriate substrate
in general; the negative result is specific to the declared readout, the rate model, and the
operating points scanned.

**Growth is not the same problem as capacity sharing.** Specialised small circuits handle single
tasks, and Cadence's own connectome lane reports that a reduced graph matched on population means
did not match on member behaviour. Retaining transfer while growing requires shared structure,
which is where interference appears. The evidence gate and the outcome gate bound the damage of a
wrong growth but do not solve sharing.

**Value comes from somewhere.** A reward channel needs a scale. A scale supplied by the task caps
the system at that task; a scale shaped by selection is endogenous but drifts toward whatever is
measured. Both are legitimate designs and they must be declared, because the choice determines
which measurements are meaningful. The systems that avoid a human score entirely, by making
survival and energy the only currency, are the only ones in this space whose objective is not
supplied by a person.

**Rule families do not mix.** A directed measured connectome does not satisfy the equilibrium
contrast gradient contract. Structural growth on directed wiring therefore belongs with a
predictive-coding or inference-learning rule, or it must accept that its updates are heuristic.
Mixing the two under one set of convergence assumptions is not a defensible default.

# 7 Reproducibility

Every table above is produced by one file in `probes/`, and `ep4nn/grow.py` runs its own
assertions. The environment is a virtual environment with NumPy and SciPy installed from a
mirror; the library is a local checkout with `PYTHONPATH` set. The worm connectome is committed
with the repository and its SHA-256 recorded. No GPU, no external service and no wallet is
involved. Measured numbers depend on the host, so the receipts record the operation counts
alongside wall-clock times.

# 8 Conclusion

A measured connectome gives structure cheaply and dynamics not at all. The useful question for a
system that grows is not how many neurons it has but whether the answer it produces separates the
situations that must be separated. ep4nn encodes that question as a prerequisite, measures it
before it grows, and records a refusal to grow as a result in its own right. On the 302-neuron
worm graph the prerequisite fails, and reporting that failure is more useful than adding
structure to a code that carries nothing. The contract and the receipt format are the parts we
expect to transfer to the next substrate, in particular to connectomes whose motor neurons exist
and whose dynamics are heterogeneous.
