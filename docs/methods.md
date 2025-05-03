# Target Value Optimization Methods

This document describes three approaches for optimizing target values during neural network training, focusing on adjusting the **class** and **non-class** values rather than keeping them fixed.

---

## 1. Manual Selection

Use static values based on the activation function or intuition. Common examples:

- Class value = `1`, Non-class value = `0`
- Class value = `0.8`, Non-class value = `0.2` (for smoother gradient flow with sigmoid)

This is simple and readable but may not yield optimal training behavior.

---

## 2. Pretraining-Based Estimation

Steps:

1. Run the training data through the **untrained** network.
2. Choose an initial class value using a heuristic (e.g., highest output value).
3. Set the non-class value as the average of all other output values.

Then, during training:
- Each epoch, average the previous class value guess with the closest actual class output.
- Similarly, average the non-class value across the current non-class outputs and the previous non-class guess.

This allows the network to gradually refine what it "expects" as good output values.

---

## 3. Adaptive Refinement (Base Version)

Each epoch:
- Record all outputs.
- Adjust the class value by averaging it with the most consistent class output seen so far.
- Adjust the non-class value by averaging all non-class outputs across the epoch (and previous guesses).

This method dynamically adapts the target values to match the evolving output behavior of the network.

---

📄 For advanced variations like `εₘᵢₙ`-, `dₘₐₓ`-, and `σ`-adaptations, see [`/docs/adaptations.md`](./adaptations.md).
