# Local function approximation (LFA) framework

This repository contains code to reproduce results in our NeurIPS 2022 publication ["Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations"](https://arxiv.org/abs/2206.01254). 


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

If this work was useful for your research, please consider citing the paper.

```
@inproceedings{lfa2022,
    title={Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post hoc Explanations},
    author={Han, Tessa, Srinivas, Suraj, and Lakkaraju, Himabindu},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```

