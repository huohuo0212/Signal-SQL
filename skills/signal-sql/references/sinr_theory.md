# Signal Detection Theory & SINR Analytic Formulations

This document details the mathematical foundation of **Signal-SQL**, providing theoretical justification for Covariance Pre-whitening, CFAR detection, and the analytic SINR reward function.

---

## 1. Covariance Pre-whitening Filter (预白化滤波)

In dense text embeddings, semantic representations suffer from **anisotropy** (narrow cones in vector space) and high-frequency common linguistic background noise. To eliminate this colored noise:

1. **Noise Covariance Estimation**:
   $$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$
   $$C = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T + \epsilon I$$
   *(where $\epsilon = 10^{-5}$ is the regularization factor for matrix invertibility).*

2. **Fractional Matrix Power**:
   $$W = C^{-1/2}$$

3. **White Space Transformation**:
   $$z = (x - \mu) W$$
   In this white space, the covariance matrix becomes the identity matrix $I$. Computing cosine similarity in white space is equivalent to calculating the **Mahalanobis Distance**, completely removing correlations across feature dimensions.

---

## 2. CFAR Clutter Edge Calibration (恒虚警率门限标定)

To establish an adaptive detection threshold without setting brittle static similarity values:
1. Conduct cross-domain hard negative mining (excluding same-database examples).
2. Measure the local peak response against the background clutter window:
   $$T(x) = \frac{R_{\text{peak}} - \mu_{\text{local}}}{\sigma_{\text{local}}}$$
3. Calibrate the threshold $\Gamma_{\text{th}}$ at the $(1 - P_{\text{fa}})$ percentile of the $H_0$ noise distribution (robustly bounded between $9.0$ and $13.0$).

---

## 3. Analytic SINR Formulation (信干噪比解析奖励)

To evaluate an example portfolio $S = \{e_1, e_2, \dots, e_k\}$ for target query $x$ without invoking costly LLM rollouts:

$$R(S) = \frac{E_{\text{match}}(S, x)}{1 + \lambda_1 E_{\text{noise}}(S, x) + \lambda_2 E_{\text{inter}}(S)}$$

### A. Signal Energy $E_{\text{match}}$ (Logical Coverage)
Measures the proportion of target SQL structural features covered by the union of candidate features:
$$E_{\text{match}}(S, x) = \frac{|\text{Feat}(x) \cap (\bigcup_{e \in S} \text{Feat}(e))|}{|\text{Feat}(x)|}$$

### B. Noise Energy $E_{\text{noise}}$ (Redundant Logic Penalty)
Measures extraneous SQL logic present in candidates that is unneeded by target $x$:
$$E_{\text{noise}}(S, x) = \frac{\sum_{e \in S} |\text{Feat}(e) - \text{Feat}(x)|}{\sum_{e \in S} |\text{Feat}(e)|}$$

### C. Interference Energy $E_{\text{inter}}$ (Intra-Portfolio Diversity & Style Conflict)
Measures pairwise redundancy and formatting conflicts across chosen examples:
$$E_{\text{inter}}(S) = \frac{1}{\binom{n}{2}} \sum_{i<j} \left( \text{Sim}_{\cos}(e_i, e_j) \cdot \mathbb{I}[\text{Sim}_{\cos} > 0.9] + \frac{\text{StyleDiff}(e_i, e_j)}{D_{\text{style}}} \right)$$

*Style dimensions checked:*
- Table alias convention (`AS T1` vs full table names)
- Quote preference (`"` vs `'`)
- Keyword casing (`SELECT` vs `select`)
- Inequality operator (`!=` vs `<>`)
