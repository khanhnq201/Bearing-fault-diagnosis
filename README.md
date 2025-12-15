# Fault Frequency–Aligned Bearing Fault Diagnosis

## Introduction

Rolling element bearings are critical components in rotating machinery, directly affecting operational stability, reliability, and maintenance cost. Empirical studies show that nearly 50% of motor failures are related to bearing faults. Early and reliable fault diagnosis is therefore essential to prevent unexpected downtime and reduce maintenance expenses.

With the rapid adoption of machine learning and deep learning in condition monitoring, many bearing fault diagnosis models have reported extremely high accuracy on benchmark datasets such as CWRU. However, recent studies have revealed that these results are often compromised by **data leakage**, where signals from the same physical bearing appear in both training and testing sets. In such cases, models tend to learn experiment-specific characteristics rather than the underlying physical signatures of bearing faults, leading to poor generalization in real-world deployments.

In addition to evaluation reliability, **model interpretability** has emerged as a key requirement for industrial adoption. A trustworthy diagnostic model should not only provide high classification accuracy, but also demonstrate that its decisions are grounded in **physically meaningful fault frequencies** (e.g., BPFI, BPFO and their harmonics), which are well understood by maintenance engineers.

This project addresses both challenges by proposing a bearing fault diagnosis framework that:
- Adopts a **bearing-wise data splitting strategy** to eliminate data leakage and ensure fair, realistic evaluation.
- Focuses on **frequency-domain signal representations** that explicitly relate to characteristic fault frequencies.
- Utilizes a **1D convolutional neural network (1D-CNN)** to preserve spectral structure while enabling direct interpretation.
- Introduces a **frequency-alignment–based interpretability analysis**, quantifying how well the model’s attention aligns with known fault frequencies rather than spurious experimental artifacts.

Experiments are conducted on the Case Western Reserve University (CWRU) bearing dataset under multiple preprocessing pipelines (Raw, FFT, DWT, Order FFT). The results demonstrate the trade-off between accuracy and reliability under different data-splitting protocols, and provide evidence of whether the model truly learns physically meaningful fault characteristics.

Overall, this project aims to move beyond accuracy-centric evaluation and contribute toward **reliable, interpretable, and physically grounded bearing fault diagnosis** suitable for real industrial applications.
