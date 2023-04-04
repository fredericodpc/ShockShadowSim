# Simulation of Observable In-Flight Shockwave Shadows
---
## About
This repository contains parts of the research work performed by Dr Frederico Paulino for his PhD studies at University of Bristol, Bristol UK, as part of a Industrial Cooperative Awards in Science & Technology (iCASE) project sponsored by Airbus and the Engineering and Physical Sciences Research Council (EPSRC). The following description is extracted from the thesis itself [1].

[PhD Thesis](https://research-information.bris.ac.uk/en/studentTheses/simulation-of-observable-in-flight-shockwave-shadows)

## Introduction
A *shock wave* occurs when matter, irrespective of the state, undergoes a rapid compression. Shock waves delimit a three-dimensional boundary across which an abrupt variation of the thermodynamic properties of the medium is observed, including density. This discontinuity propagates in space with supersonic speeds.

Settles [2] noted that a typical shock wave encountered in compressible flow, travelling just over the sonic speed, induces a very small variation of around $10^{-5}$ in the refractive index of air due to its characteristic density gradient. This inhomogeneity in the optical medium in turn causes the refraction of light and a deflection of only approximately $10^{-3}$ degrees or less [2]. Although such a weak discontinuity and small light bending are usually invisible to the naked eye, the light displacement, deviation and interference resulting from the interaction with compressible flow features can be very pronounced. The outcome is that image features associate with shock waves can occur naturally, when the conditions are right and 'one knows where and how to look' [2]. 

The diverging high intensity light that is emitted from the sun, which after travelling a very long distance arrives at the earth as a parallel beam, can serve as an appropriate light source to generate direct natural shadowgraphs. These shadowgraphs of a refractive index field can be cast on any approximately flat and diffuse surface [2]. These days airline passengers can often see this shadow and caustic (envelope of refracted rays [3]) pattern [4] over the wing when the Sun is approximately overhead [2]. However, what is visible from the airborne perspective is completely dependent on the particular vehicle operation, illumination, observer and recording conditions.

![alt text](https://github.com/fredericodpc/ShockShadowSim/blob/main/research/figures/natural_shockshadow.svg "Natural Shockshadow")

## Motivation

Direct natural sunlight shadowgraphy is therefore an interesting and potential valuable subject for study. The shock wave shadows or "shockshadow" phenomenon presents itself as a possible alternative method to the study of transonic and compressible ﬂow in general [5]. Knowledge of the conditions under which the shockshadow becomes a visible element may lead to the conception of a technique that exploits the pattern as an additional source of information about the ﬂow feature(s) that originated it [5]. This new information can then be used to improve old or possibly provide completely new insights into the physical processes of compressible flow, the very definition of the role of a fluid flow visualisation technique [6].

A computationally simulated shockshadow can then be used to:
+ Predict its observation
+ Design experiments around this particular phenomenon
+ Further evaluate its properties
+ Create entirely new strategies to visualise compressibility phenomena [1]
+ Portray a single or a whole system of shock waves on surfaces of aircraft in the transonic flight regime and provide indication of their locations [7, 8]
+ Detect these flow features, i.e. obtain a practical representation of the flow [7]
+ Track, measure and characterise the shock wave system; detect shock wave-induced flow separation; monitor aeroelastic instabilities through the shock wave osciallatory behaviour [9]
+ Validate analytical solutions; verify and validate computational fluid dynamics solvers; wind tunnel/flight testing correlation or validation of experiments [7]

## Light Simulation Framework

![alt text](https://github.com/fredericodpc/ShockShadowSim/blob/main/research/figures/light_simulation_framework.svg "Light Simulation Framework")

Module | Model(s)
--- | ---
Fluid Flow | Transonic flow over the Onera M6-Wing with OpenFoam
Light source | Planck's Blackbody Radiation equation + sRGB color space + wing projection (azimuth & elevation)
Light propagation | Ray Equation 
Light refraction | DOP853 Runge-Kutta + Shape functions gradient reconstruction + Cell location-based interpolation
Light reflection | Lambertian material  
Viewing | Virtual pinhole camera
Image synthesis | Rendering equation + Photon Mapping + Epanechnikov Kernel

![alt text](https://github.com/fredericodpc/ShockShadowSim/blob/main/research/figures/onera_m6_cfd.svg "Onera M6-Wing CFD")
![alt text](https://github.com/fredericodpc/ShockShadowSim/blob/main/research/figures/synthetic_shockshadow.svg "Synthetic Shockshadow")

## Running instructions

External libraries:
- NumPy
- SciPy
- PyVista (VTK wrapper)
- VTK

```
python shockshadow_illumination_generator.py
```

UNDER CONSTRUCTION...


## Resources
[1] F. D. P. Costa, “Simulation of observable in-flight shockwave shadows”, Ph.D. thesis, University of Bristol, 2022.

[2] G. S. Settles. Schlieren and shadowgraph techniques: visualizing phenomena in transparent media. Springer Science & Business Media, 2001.

[3] E. Hecht. Optics, 5e. Pearson Education India, 2017.

[4] D. F. Fisher, E. A. Haering, G. K. Noffz, and J. I. Aguilar. Determination of sun angles for observations of shock waves on a transport aircraft. - - NASA/TM-1998-206551, 1998.

[5] C. L. Johnson. Development of the Lockheed P-80A jet fighter airplane. Journal of the Aeronautical Sciences, 14(12):659–679, 1947.

[6] W. Merzkirch. Flow visualization. Elsevier, 2012.

[7] J. R. Crowder. Flow visualization techniques applied to full-scale vehicles. AIAA Atmospheric Flight Mechanics Conference, AIAA-1987-2421, 1987.

[8] T. M. Tauer, D. L. Kunz, and N. J. Lindsley. Visualization of nonlinear aerodynamic phenomena during F-16 limit-cycle oscillations. Journal of Aircraft, 53(3):865–870, 2016.

[9] G. A. Rathert and G. E. Cooper. Visual observations of the shock wave in flight. NACA-RM-A8C25, 1948.

















