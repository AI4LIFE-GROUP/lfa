# Local function approximation (LFA) framework

This repository contains code to reproduce results in our NeurIPS 2022 publication ["Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations"](https://arxiv.org/abs/2206.01254). 


## Summary

Under the local function approximation (LFA) framework, explanations perform local function approximation of a complex model over a local neighbourhood using a simple model based on a loss function. The LFA framework unifies eight diverse popular post hoc explanation methods (i.e., LIME, C-LIME, KernelSHAP, Occlusion, Vanilla Gradients, SmoothGrad, Gradient x Input, and Integrated Gradients). Using the LFA framework, we show that no single explanation method can perform optimally over every local neighbourhood, calling for a principle approach to select among methods. To select among methods, we set forth a guiding principle, deeming a method to be effective if it performs faithful LFA. Using the LFA framework, we determine the conditions under which each existing explanation methods are effective. If, in a given situation, no existing method is effective, the LFA framework also provides a way to design novel methods (by specifying an appropriate model class, local neighborhood, and loss function) that are tailored to the given situation and that satisfy the guiding principle.


## Usage

To reproduce results, navigate into the repository follow the steps below.

**1. Generate explanations for individual model predictions**
   
   * Run `$ python experiments/generate_explanations.py`.
   * Explanations are generated to explain the individual model predictions of each model (four regression models and four classification models) using each explanation method (LIME, KernelSHAP, Occlusion, Vanilla Gradients, Gradient x Input, SmoothGrad, and Integrated Gradients). Each explanation is computed using two approaches: the existing approach (implemented by the Captum library) and the LFA framework (implemented in the `lfa` folder).
   * Explanations use 1000 perturbations per data point. If running on a local machine, use a smaller number of perturbations by changing line 165 (`n_perturbs_list = [1000]`) in `experiments/generate_explanations.py`
   * Explanations are saved in `experiments/results`.

**2. Analyze explanations**
   
   * Run `$ python analysis/analyze_explanations.py`.
   * Figures are saved in `analysis/figures`. These are the figures that appear in the paper.


## Citation

```
@inproceedings{lfa2022,
    title={Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations},
    author={Han, Tessa, and Srinivas, Suraj, and Lakkaraju, Himabindu},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```

