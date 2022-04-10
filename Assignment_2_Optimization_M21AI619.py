# THIS IS PROJECT FOR OPTIMIZATION IN DATA SCIENCE
# PREPARED BY Tathagata Mookherjee
# IIT JODHPUR
# ROLL NUMBER - M21AI619
# DATE : 11/12/2021
# Comments: Implementation using bespoke program which validates armijo-wolfe condition inside a while loop

import sympy as sympy
import numpy as np

#############################
#############################
# enter roll number here
roll = 'M21AI619'
#############################
#############################

# declaring variables and initial data provided
val_roll = float(roll[6: 8])
x, y = sympy.symbols('x y')
# sympy.sympify is not able to handle floats, hence val_roll converted to int
fx_formula = ((x-int(val_roll))**2)+((y-(2*x))**2)  #provided in question
fx_eq = sympy.sympify(fx_formula)
delfx_eq = [sympy.diff(fx_formula, x), sympy.diff(fx_formula, y)]
epsilon = np.array(0.001).astype(np.float64)  # 10^-3 provided in question
alpha_1 = 1.0  #assumed
r = 0.5  #assumed
beta_1 = 0.0001  # 10^-4  #assumed
beta_2 = 0.9  #assumed
val_x_initial = [val_roll+3, (2*val_roll)-2]  #provided in question
k = 1  #iteration counter

print('roll: '+str(roll))
print('val_roll: '+str(val_roll))
print('fx_formula: '+str(fx_formula))
print('fx_eq: '+str(fx_eq))
print('delfx_eq: '+str(delfx_eq))
print('epsilon: '+str(epsilon))
print('alpha_1: '+str(alpha_1))
print('r: '+str(r))
print('beta_1: '+str(beta_1))
print('beta_2: '+str(beta_2))
print('val_x_initial: '+str(val_x_initial))
print('k: '+str(k))

# function to get value for any X from FX
def get_fx_val(val_x):
    # get value of fx after passing x,y
    fx_val_x = fx_eq.subs([(x, val_x[0]), (y, val_x[1])])
    # convert value from int to float and return
    return np.array(fx_val_x).astype(np.float64)

# function to get value for any X from delFX
def get_delfx_val(val_x):
    # get value of delfx after passing x,y
    delfx_val_x = [delfx_eq[0].subs([(x, val_x[0]), (y, val_x[1])]), delfx_eq[1].subs([
        (x, val_x[0]), (y, val_x[1])])]
    # convert value from int to float and return
    return np.array(delfx_val_x).astype(np.float64)


# main program starting...
print('NOTE: checking to see if initial val_x is the local minima of fx..')
del_fx_val = get_delfx_val(val_x_initial)  # get value for delfx(x0)
norm_delfx_val = np.linalg.norm(del_fx_val)  # get norm of value for delfx(x0)

# checking to see if norm of value for delfx(x0) > epsilon
if(norm_delfx_val < epsilon):
    print('Answer: norm of del-F(X0): '+str(norm_delfx_val)+' <= epsilon: ' +
          str(epsilon)+'. Hence exiting as X0 is the local minima of F(X)..')
    exit(1)
else:
    print('NOTE: norm_delfx_val>epsilon, so checking armijo-wolfe..')

# armijo-wolfe starts here..
alpha_itr = alpha_1
valx_itr = val_x_initial
val_d_itr = -1*get_delfx_val(val_x_initial)
while (norm_delfx_val >= epsilon):
    print('NOTE: checking armijo with below..')
    print('alpha_itr: '+str(alpha_itr))
    print('valx_itr: '+str(valx_itr))
    print('val_d_itr: '+str(val_d_itr))

    fx_val = get_fx_val(valx_itr)  # value of fx_eq with values from valx_itr
    print('fx_val: '+str(fx_val))
    alpha1_beta1_delfxT_d = alpha_1*beta_1 * \
        np.dot(get_delfx_val(valx_itr), val_d_itr)
    print('alpha1_beta1_delfxT_d: '+str(alpha1_beta1_delfxT_d))
    armijo_check = fx_val+alpha1_beta1_delfxT_d
    print('armijo_check: '+str(armijo_check))

    val_xalphad_itr = valx_itr+alpha_itr*val_d_itr  # x+a*d
    print('val_xalphad_itr: '+str(val_xalphad_itr))
    armijo_check_val_fx_xalphad_itr = get_fx_val(val_xalphad_itr)  # F(x+a*d)
    print('armijo_check_val_fx_xalphad_itr: ' +
          str(armijo_check_val_fx_xalphad_itr))
    if(armijo_check >= armijo_check_val_fx_xalphad_itr):
        print('NOTE: armijo passed checking wolfe for below..')
        print('alpha_itr: '+str(alpha_itr))
        print('valx_itr: '+str(valx_itr))
        print('val_d_itr: '+str(val_d_itr))
        # wolfe check condition
        del_fx_val_itr = get_delfx_val(valx_itr)  # delfx(x)
        wolfe_check = beta_2*np.dot(del_fx_val_itr, val_d_itr)
        wolfe_check_val_delfxd = np.dot(
            get_delfx_val(val_xalphad_itr), val_d_itr)
        print('wolfe_check: '+str(wolfe_check))
        print('wolfe_check_val_delfxd: '+str(wolfe_check_val_delfxd))
        if(wolfe_check_val_delfxd >= wolfe_check):
            print('NOTE: wolfe passed for below..')
            print('ANS: (xK+alpha*dK): '+str(valx_itr))
            print('ANS: (alpha step length): '+str(alpha_itr))
            print('ANS: (dk descent direction): '+str(val_d_itr))
            valx_itr = val_xalphad_itr  # setting x=x+alpha*d
            alpha_itr = alpha_1  # a=1
            del_fx_val_itr = get_delfx_val(valx_itr)  # delfx(x)
            val_d_itr = -1*get_delfx_val(valx_itr)  # d=-delfx(x)
            # get norm of value for delfx(x)
            norm_delfx_val = np.linalg.norm(del_fx_val_itr)
            print(
                'NOTE: updating norm_delfx_val to check if exit criteria has been met..')
            print('del_fx_val_itr: '+str(del_fx_val_itr))
            print('norm_delfx_val: '+str(norm_delfx_val))
            print('epsilon: '+str(epsilon))
        else:
            print('NOTE: wolfe failed, updating below..')
            alpha_itr = alpha_itr*r
            print('alpha_itr: '+str(alpha_itr))
    else:
        print('NOTE: armijo failed, updating below..')
        alpha_itr = alpha_itr*r
        print('alpha_itr: '+str(alpha_itr))
    k = k + 1

print('norm_delfx_val>epsilon criteria has been met. exiting..')
print('k: '+str(k))
