# Adversarial Training through the Problem-Space of Dynamic Analysis

A Reinforcement Learning based framework for making dynamic analysis malware classifiers robust to evasive attacks.
In the typical minmax fashion of adversarial training, the RL agent replaces approaches like FGSM and PGD in the inner maximization loop for 3 main reasons.
It performs the set of modifications that are feasible in the problem-space, and only those.
Thus it completely circumvent the inverse mapping problem.
It simplifies the computational task as gradient-based perturbations are often intractable.
Finally, in this manner we can provide theoretical guarantees for the robustness of the model against this particular set of adversarial actions in the problem-space.

## Requirements

- Python 3.8.10
- Julia 1.6.3
- PyJulia 0.6.1
- OpenAI-Gym 0.21.0
- Stable-Baselines-3 1.7.0
