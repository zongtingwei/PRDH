<div align="center">
<h1 align="center">
</h1>
<h1 align="center">
PRDH: code for "Solving Multiobjective Feature Selection Problems in Classification via Problem Reformulation and Duplication Handling"
</h1>

[![Platform](https://img.shields.io/badge/Platform-MATLAB-orange)](https://www.mathworks.com/products/matlab.html)
[![Dataset](https://img.shields.io/badge/Dataset-feature_selection-green)](https://github.com/zongtingwei/Feature-Selection-FS-datasets)

[Source Code](https://github.com/zongtingwei/PRDH)
| [Documentation](https://ieeexplore.ieee.org/abstract/document/9925116)
| [Dataset](https://github.com/zongtingwei/Feature-Selection-FS-datasets)

</div>
<br>

## 📖 Introduction

PRDH is a MATLAB-based evolutionary algorithm designed for solving multiobjective feature selection problems in classification tasks. It leverages problem reformulation and duplication handling to enhance the efficiency and effectiveness of the feature selection process.

This implementation is based on the code of [SM-MOEA](https://github.com/BIMK/SM-MOEA) and [PlatEMO](https://github.com/BIMK/PlatEMO). Please refer to the original paper [Solving Multiobjective Feature Selection Problems in Classification via Problem Reformulation and Duplication Handling](https://ieeexplore.ieee.org/abstract/document/9925116) for detailed information about the algorithm's overview, methodology, and benchmark results.

This code was developed for feature selection tasks in classification. The framework can be adapted to other feature selection scenarios with minor modifications.

## 🔥 News

+ 🎉🎉 Coming soon

## 💡 Features of our package

| Feature | Support / To be supported |
|---------|---------------------------|
| **Efficient Feature Selection** | 🔥Support |
| **Multi-Objective Optimization** | 🔥Support |
| **Classification Task Support** | 🔥Support |
| **MATLAB Implementation** | 🔥Support |
| **High-Dimensional Data Support** | 🔥Support |
| **More Application Scenarios** | 🚀Coming soon |

## 🎁 Requirements & Installation

> [!Important]
> This implementation requires MATLAB. Ensure you have MATLAB installed on your system.

> [!Note]
> The code is based on SM-MOEA and PlatEMO. Please download the required libraries if necessary.

### How to Run

1. Download the code and dataset from the repository.
2. Open MATLAB and set the working directory to the project root.
3. Run the `main_prdh.m` script.
4. You can choose the provided "colon.mat" file in the "dataset" folder for testing.

```matlab
% an example
% you can find the code in `main_prdh.m` file
algorithmName = 'PRDH';  
dataNameArray = {'colon'}; % dataset
global maxFES
maxFES = 100;  % max number of iteration
global choice
choice = 0.6; % the threshold choose features
global sizep
sizep = 300; % size of population
```
## ⚙️ References
Jiao R, Xue B, Zhang M. [Solving Multiobjective Feature Selection Problems in Classification via Problem Reformulation and Duplication Handling](https://ieeexplore.ieee.org/abstract/document/9925116)[J]. IEEE Transactions on Evolutionary Computation, 2022.
###
Tian Y, Cheng R, Zhang X, et al. [PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective Optimization [Educational Forum]](https://ieeexplore.ieee.org/abstract/document/8065138   )[J]. IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87.
###
Cheng F, Chu F, Xu Y, et al. [A Steering-Matrix-Based Multiobjective Evolutionary Algorithm for High-Dimensional Feature Selection](https://ieeexplore.ieee.org/abstract/document/9371430   )[J]. IEEE transactions on cybernetics, 2021, 52(9): 9695-9708.
###

## 🪪 License
This project is based on the implementation of [SM-MOEA](https://github.com/BIMK/SM-MOEA) and [PlatEMO](https://github.com/BIMK/PlatEMO). Please refer to their respective licenses for details.

## ☎️ Contact
If you encounter any issues or have questions regarding PRDH, please feel free to contact me.

## ⭐ Star
If you find this work helpful, please consider giving me a ⭐!
