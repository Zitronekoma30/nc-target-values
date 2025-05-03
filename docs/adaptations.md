# Adaptation Variants for Target Value Optimization

These adaptations modify how class and non-class values are refined during training, with added constraints or heuristics to improve stability and separation.

---

## 🔸 `dₘₐₓ`-Adaptation

This variant simplifies non-class value selection using a threshold.

**Rules:**
- If class value > `0.5` → non-class = `0`
- If class value ≤ `0.5` → non-class = `1`
- Enforce all values to remain within `[0, 1]`.

**Parameter:**
- `dₘₐₓ` defines how far apart class and non-class values should be, though its exact role is underspecified. It helps maintain separation and training stability.

---

## 🔸 `εₘᵢₙ`-Adaptation

Ensures a **minimum distance** between class and non-class values.

**Rules:**
- Let `εₘᵢₙ ∈ [0,1]`
- If `|class - non-class| < εₘᵢₙ`:
  - Try adjusting non-class:
    - Up: `non-class + εₘᵢₙ`
    - Down: `non-class - εₘᵢₙ`
  - Choose the option that keeps the value within `[0, 1]`
  - If both would violate bounds, choose the direction that increases separation the most

This guards against class/non-class collapse.

---

## 🔸 `σ`-Adaptation

Uses the variability in outputs (standard deviation) to dynamically enforce separation.

**Steps:**
1. Compute the sum of standard deviations for current outputs: `σ_sum`
2. Ensure:
   - All values remain within `[0, 1]`
   - `|class - non-class| ≥ σ_sum / 2`
3. If too close:
   - Move class **up** by `σ_sum / 2`
   - Move non-class **down** by `σ_sum / 2`
4. If this would exceed bounds, only move the safe value or clip the other to `0` or `1`.

This strategy adapts the separation to match output uncertainty.

---

These adaptations attempt to improve convergence, stability, and separation between outputs but can introduce additional hyperparameters and complexity.
