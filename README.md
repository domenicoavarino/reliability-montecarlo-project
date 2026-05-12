# Reliability Monte Carlo Project

Monte Carlo simulator for reliability and availability analysis of fault-tolerant systems using event-driven simulation.

The project is developed as a university project and is based on stochastic modeling, exponential failure/repair processes, and discrete-event simulation techniques.

The simulator evolves incrementally through multiple project points, progressively adding:

* redundancy logic,
* repair mechanisms,
* queues,
* external shocks,
* availability estimation,
* advanced event handling.

---

# ✨ Main Features

## 🎲 Monte Carlo Simulation

Randomized simulation of system evolution through probabilistic event generation.

## ⏱️ Event-Driven Engine

The simulator does NOT evolve second-by-second.

Instead, time jumps directly:

* from one event to the next,
* improving efficiency and scalability.

## ⚡ Exponential Event Sampling

Generation of:

* failure times,
* repair times,
* external events

using exponential distributions.

Core formula:

T = -ln(1-U)/λ

where:

* U is a random uniform variable,
* λ is the event rate.

## 🔧 Reliability Modeling

Support for:

* series systems,
* parallel systems,
* k-out-of-n systems,
* redundant architectures.

## 🛠️ Repair Logic

Simulation of:

* repair processes,
* repair queues,
* maintenance behavior.

## 📊 Statistics & Metrics

Estimation of:

* reliability,
* availability,
* downtime,
* failure probabilities,
* steady-state metrics.

---

# 🧠 Core Concepts Used

The project is based on:

* Monte Carlo methods
* Event-driven simulation
* Exponential distributions
* Reliability theory
* Availability analysis
* Markovian concepts
* Stochastic processes

---

# 🛠️ Technologies Used

Language:

* Python 3.10+

Libraries:

* random
* math
* numpy (future)
* matplotlib (future)

Development Environment:

* VS Code
* GitHub
* Ubuntu / WSL (recommended)

---

# 🚀 Project Setup

## Prerequisites

Install:

* Python 3.10+
* Git
* VS Code

Recommended:

* Ubuntu / WSL on Windows

---

# 1. Clone Repository

```bash
git clone https://github.com/dodosniper98/reliability-montecarlo-project.git
cd reliability-montecarlo-project
```

---

# 2. Open Project

```bash
code .
```

---

# 3. Run Current Sampling Test

```bash
python src/common/sampling.py
```

This generates random exponential event times.

---

# 📂 Project Structure

```text
.
├── src/
│   ├── common/
│   │   └── sampling.py
│   │
│   ├── point0/
│   ├── point1/
│   ├── point2/
│   └── point3/
│
├── notebooks/
├── results/
├── report/
│
└── README.md
```

---

# 📌 Project Organization

## src/common/

Shared simulator components:

* exponential sampling,
* event queue,
* statistics,
* reusable utilities,
* component models.

## src/point0/

Initial system simulation:

* simple 2oo3 logic,
* no repair.

## src/point1/

Adds:

* subsystem logic,
* more complex reliability behavior.

## src/point2/

Adds:

* repair logic,
* availability analysis.

## src/point3/

Adds:

* repair queues,
* external shock events,
* advanced event handling.

---

# 🔄 Recommended Development Workflow

Before starting work:

```bash
git pull
```

After modifying code:

```bash
git add .
git commit -m "description"
git push
```

---

# ⚠️ Important Notes

Git does NOT track empty folders.

To preserve project structure:
`.gitkeep` files may be used.

---

# 🎯 Recommended Development Order

1. Exponential random sampling
2. Single component simulation
3. 2oo3 system simulation
4. Monte Carlo repetitions
5. Reliability estimation
6. Repair logic
7. Event queue
8. Availability estimation
9. External shock events

---

# 📚 Current Status

Implemented:

* project structure,
* GitHub synchronization,
* exponential event sampling.

Future work:

* full event engine,
* component state logic,
* repair simulation,
* statistics collection,
* availability analysis.

---

# 📝 License

University project for educational purposes.
