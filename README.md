# gradient-descent-armijo-wolfe
Bespoke, from scratch, implementation of Armijo-Wolfe inexact line search technique to find step length for gradient descent optimisation. The library alternative is scipy.optimize.line_search

Example inputs provided to program:
1.	roll = 'M21AI619'
2.	R = last 2 digit of roll dynamically obtained
3.	FX = (x1 − R)^2 + (x2 − 2x1)^2
4.	delFX = dynamically calculated from FX
5.	epsilon = 10^-3
6.	alpha_1 = 1.0
7.	r = 0.5
8.	beta_1 = 10^-4
9.	beta_2 = 0.9
10.	X0 = [R+3,2R-2]
11.	D = dynamically calculated from delFX
12.	k = 1

Example output of program:
1.	F(x+alpha*dK): [18.99935447 37.99813236]
2.	Alpha (step length): 0.125
3.	Dk (descent direction): [-0.00101529  0.00115317]
4.	K (number of iterations): 323
