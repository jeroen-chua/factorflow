This is a generic loopy belief propagatinon framework for factor graphs. This work assumes the user has some familiarity with factor graphs and loopy belief propagation. A tutorial on factor graphs can be found in:

Kschischang, Frey, Loeliger. "Factor Graphs and the Sum-Product Algorithm", IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 47, NO. 2, FEBRUARY 2001.

Bishop. "Pattern Recognition and Machine Learning", Springer-Verlag New York, Inc., Secaucus, NJ, USA, 2006.

The purpose of the framework is to provide a generic platform to perform standard and variants of belief propagation (sum-product, max-product, tree re-weighted, convergent belief propagation). The framework is flexible and comes with some pre-built factors and message-passing schemes. New factors can be used in the framework simply by defining how messages are passed out of the factors. 

The "examples" folder contains examples of using certain factors and certain message-passing schemes. The framework can be extended to new factors and message-passing schemes by subclassing the FactorNodes class and filling in three functions (see code for details).

To cite this framework, cite this github repository for now. A tech report may come in the future if there's enough interest.

- Jeroen Chua
