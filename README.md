# transitional-mcmc

This repo contains the code of Transitional Markov chain Monte Carlo algorithm. TMCMC method is a simulation-based Bayesian inference technique which sample from the
complete joint posterior distribution of the unknown parameter vector θ . In the literature, there are several closely related algorithms such as cascading adaptive transitional metropolis in parallel, sequential Monte Carlo, particle filters, bootstrap filters, condensation algorithm,
survival of the fittest and population Monte Carlo algorithms. TMCMC do not require the Gaussian assumption about the prior and posterior PDFs of the unknown parameters, an inherent assumption in Kalman filters and its nonlinear variants.

Pseudo code of the algorithm can be found in Table 2 of the following paper:

Ramancha, M. K., Astroza, R., Madarshahian, R., and Conte, J. P. (2022). “Bayesian updating and identifiability assessment of nonlinear finite element models.“ Mechanical Systems and Signal Processing, 167, 108517. https://doi.org/10.1016/j.ymssp.2021.108517

![image](https://user-images.githubusercontent.com/41924394/170327389-0c2906bb-0761-497f-b256-eb1588b704e4.png)

## main_2DOF.py

main_2DOF.py is an example scipt that solves the problem mentioned in Section 2.2.3 of the following book:

Yuen, K. V. (2010). “Bayesian Methods for Structural Dynamics and Civil Engineering.“ John Wiley & Sons, Ltd, Chichester, UK. [Link](https://civiltechnocrats.files.wordpress.com/2013/11/bayesian-methods-for-structural-dynamics-and-civil-engineering.pdf)

see [2DOF_example.pdf](2DOF_example.pdf) document

### TMCMC Samples at final stage

![image](https://user-images.githubusercontent.com/41924394/170321794-bf395669-8623-454c-9b67-4bf66feefa7b.png)

### TMCMC Samples at each stage

![ezgif com-gif-maker](https://user-images.githubusercontent.com/41924394/198080622-b50b7ea2-1765-4e34-a2e0-bacb441c75c0.gif)

## quoFEM

The TMCMC code in this repo also serves as the backend of NHERI SIMCENTER [quoFEM](https://simcenter.designsafe-ci.org/research-tools/quofem-application/) toolbox

See [TMCMC algorithm](https://nheri-simcenter.github.io/quoFEM-Documentation/common/technical_manual/desktop/UCSDUQTechnical.html) description 

See [2 story building example](https://nheri-simcenter.github.io/quoFEM-Documentation/common/user_manual/examples/desktop/qfem-0014/README.html) solved using TMCMC and quoFEM
