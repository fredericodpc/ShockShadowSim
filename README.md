# Simulation of Observable In-Flight Shockwave Shadows

A *shock wave* occurs when matter, irrespective of the state, undergoes a rapid compression. Shock waves delimit a three-dimensional boundary across which an abrupt variation of the thermodynamic properties of the medium is observed, including density. This discontinuity propagates in space with supersonic speeds.

Settles noted that a typical shock wave encountered in compressible flow, travelling just over the sonic speed, induces a very small variation of around $10^{-5}$ in the refractive index of air due to its characteristic density gradient. This inhomogeneity in the optical medium in turn causes the refraction of light and a deflection of only approximately $10^{-3}$ degrees or less \citep{Settles:2001:schlieren}. Although such a weak discontinuity and small light bending are usually invisible to the naked eye, the light displacement, deviation and interference resulting from the interaction with compressible flow features can be very pronounced. The outcome is that image features associate with shock waves can occur naturally, when the conditions are right and 'one knows where and how to look' [Settles]. 

The diverging high intensity light that is emitted from the sun, which after travelling a very long distance arrives at the earth as a parallel beam, can serve as an appropriate light source to generate direct natural shadowgraphs. These shadowgraphs of a refractive index field can be cast on any approximately flat and diffuse surface [Settles]. These days airline passengers can often see this shadow and caustic (envelope of refracted rays [Hecht]) pattern [Fisher] over the wing when the Sun is approximately overhead [Settles]. However, what is visible from the airborne perspective is completely dependent on the particular vehicle operation, illumination, observer and recording conditions.

---
## Motivation

Direct natural sunlight shadowgraphy is therefore an interesting and potential valuable subject for study. This research is in response to this fundamental question: what can be inferred about the shock wave and the compressible flow field in general from the observed shadow and caustic patterns?


The shockshadow phenomenon therefore presents itself as a possible alternative method to the study of transonic and compressible ﬂow in general \citep{Johnson:1947:development}. Knowledge of the conditions under which the shockshadow becomes a visible element may lead to the conception of a technique that exploits the pattern as an additional source of information about the ﬂow feature(s) that originated it \citep{Johnson:1947:development}. This new information can then be used to improve old or possibly provide completely new insights into the physical processes of compressible flow, the definition of the role of a fluid flow visualisation technique \citep{Merzkirch:2012:flow}.

+ Predict its observation
+ Design experiments around this particular phenomenon
+ Further evaluate its properties
+ Create entirely new strategies to visualise compressibility phenomena [1]
+ Portray a single or a whole system of shock waves on surfaces of aircraft in the transonic flight regime and provide indication of their locations [36, 38]
+ Detect these flow features, i.e. obtain a practical representation of the flow [36]
+ Tracking, measurement and characterisation of the shock wave system; detection of shock wave-induced flow separation; monitoring of aeroelastic instabilities through the shock wave osciallatory behaviour [34]
+ Validation of analytical solutions; verification and validation of computational fluid dynamics solvers; wind tunnel/flight testing correlation or validation of experiments [36]









## Light Simulation Framework
![alt text](https://github.com/fredericodpc/ShockShadowSim/blob/main/research/figures/synthetic_shockshadow.png "Test")

Module | Model(s)
--- | ---
Fluid Flow | Transonic flow over the Onera M6-Wing with OpenFoam
Light source | Planck's Blackbody Radiation equation + sRGB color space + wing projection (azimuth & elevation)
Light propagation | Ray Equation 
Light refraction | DOP853 Runge-Kutta + Shape functions gradient reconstruction + Cell location-based interpolation
Light reflection | Lambertian material  
Viewing | Virtual pinhole camera
Image synthesis | Rendering equation + Photon Mapping + Epanechnikov Kernel

## Running instructions

External libraries:
- NumPy
- SciPy
- PyVista (VTK wrapper)
- VTK
- 

UNDER CONSTRUCTION...


## Resources
[1]
[2]
[3]
[4]
[5]
[6]
[7]
[8]


## About
This repository contains parts of the research work performed by Dr Frederico Paulino for his PhD thesis at University of Bristol, a Industrial Cooperative Awards in Science & Technology (iCASE) project sponsored by the Engineering and Physical Sciences Research Council (EPSRC) and Airbus.














