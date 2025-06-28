## 🎯 Target Value Optimization in Neural Networks

When training a classification neural network, each output neuron represents a class. For example, in a 3-class problem (`cat`, `dog`, `rat`), a typical target vector for a `cat` image might be:

![image](https://github.com/user-attachments/assets/a8c45295-7bd0-41dc-90fa-63855b123508)

These values (class = 1, non-class = 0) are compared to the network's output to compute the loss and update the weights.

However, using fixed values like `1` and `0` may not always yield optimal learning. Adjusting the **class** and **non-class** target values (e.g., `0.8` and `0.2`) can improve training performance—especially with activation functions like sigmoid—by enhancing gradient flow.

This project explores **dynamically optimizing** these values during training instead of using fixed constants. Early results show improvements in training speed and confidence, though gains in test accuracy are yet to be achieved.

📄 See [`/docs`](./docs) for more details.

## ✅ TO-DO

### 🧩 Implementation
- [X] Implement basic σ-adaptation logic (only nc)
- [X] Switch to using one non-class value for each class instead of one global one
- [X] Fix confidence calculation: Calculate cosine similarity of output vector to closest target vector (not necessarily the correct target)
- [X] Log current class/non-class values
- [X] Set up structured experiments and logging
- [X] Build simple frontend for selecting method, starting and stopping training and displaying result in graph (maybe use "tensorboard" or "weights and biases")
- [ ] Find new ways of initializing class and non-class values
  - [X] Uniform init (poor performance)
  - [ ] Soft target init
  - [X] Implement initial nudge for network preference init
  - [ ] Try taking all the initial preferences and remapping them to use the full (0,1) range
- [ ] Try pushing after each batch 

### 🧠 Algorithm Design
- [X] Refine the σ-adaptation strategy (tuning, edge cases)
- [X] Explore and prototype additional adaptation methods

### 🔬 Research & Evaluation
- [X] Define evaluation metrics beyond accuracy/loss (e.g. training speed, confidence margin)
- [X] Analyze, visualize, and summarize results
- [X] Draft and outline the research paper
- [ ] Finish paper
