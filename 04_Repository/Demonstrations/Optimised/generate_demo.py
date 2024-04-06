import numpy as np
import sympy as sym

# install opty: https://github.com/csu-hmc/opty 
from opty.direct_collocation import Problem
from opty.utils import parse_free
from collections import OrderedDict
import matplotlib.pyplot as plt
import time


def calculate_optimal_demo(starting_conditions, ending_conditions, duration=2.5):
    """
    Calculates energy efficient trajectory from a given starting and ending position.
    Must provide x, y values for both starting and ending conditions and a duration.
    This function operated at 100Hz i.e. a duration of 2.5 will give 250 output points.
    """
    start = time.time()
    L, m_q, m_h, g, k, I, t = sym.symbols('L, m_q, m_h, g, k, I, t')
    x1, d_x1, x2, d_x2, gamma, d_gamma, theta, d_theta, l, d_l, U, M = sym.symbols('x1, d_x1, x2, d_x2, gamma, d_gamma, theta, d_theta, l, d_l, U, M', cls=sym.Function)

    state_symbols = (x1(t), x2(t), gamma(t), theta(t), l(t), d_x1(t), d_x2(t), d_gamma(t), d_theta(t), d_l(t))
    constant_symbols = (L, m_h, m_q, g, k, I)
    specified_symbols = (U(t),M(t))
    pi = 3.14159

    #BELOW ARE THE USER ADJUSTABLE PROPERTIES
    num_nodes = (int) (duration * 100)

    # Specify the known system parameters.
    par_map = OrderedDict()
    par_map[m_q] = 5.0
    par_map[m_h] = 0.1
    par_map[L] = 0.5
    par_map[g] = 9.81
    par_map[k] = 10000.0
    par_map[I] = 10.0

    u_limit = 1000.0
    #+- physical space around origin
    x_limit = 5

    # Specify the symbolic instance constraints, i.e. initial and end
    # conditions. Ensure they are floats i.e. 0.0
    initial_conditions = np.matrix([[starting_conditions["x"], ending_conditions["x"]],         #x
                                    [starting_conditions["y"], ending_conditions["y"]],         #y
                                    [0.0,0.0],          #gamma
                                    [0.0, 0.0],       #theta
                                    [0.5,0.5],          #l - keep this fixed
                                    [0.0,0.0],          #d_x TODO: Adjust these so we have speed at the end
                                    [0.0,0.0],          #d_y
                                    [0.0,0.0],          #d_gamma
                                    [0.0,0.0],          #d_theta
                                    [0.0,0.0]])         #d_l

    #set the bounds for each variable
    #TODO: does IPOPT perform better when all variables are bounded (like SNOPT?)
    #INFO: optimisation time went from 20 sec to 1 sec just by adding constraint on theta
    bounds = {
                U(t): (0, u_limit),
                M(t): (-u_limit, u_limit),
                l(t): (par_map[L],1.1*par_map[L]),
                x1(t): (-x_limit, x_limit),
                x2(t): (par_map[L],x_limit),
                theta(t): (0.0,pi)
                }

    #ALL USER INPUTS ARE ABOVE THIS LINE. DO NOT MODIFY BELOW

    interval_value = duration / (num_nodes - 1)

    #these are populated in the format q(t) - q_t = 0 form the 'initial_conditions' matrix above
    #want to use np.mod(,2pi) on each of the angle constraints but cant.
    instance_constraints = (x1(0.0) - initial_conditions[0,0],
                            x1(duration) - initial_conditions[0,1],
                            x2(0.0) - initial_conditions[1,0],
                            x2(duration) - initial_conditions[1,1],
                            gamma(0.0) - initial_conditions[2,0],
                            gamma(duration) - initial_conditions[2,1],
                            theta(0.0) - initial_conditions[3,0],
                            theta(duration) - initial_conditions[3,1],
                            l(0.0) - initial_conditions[4,0],
                            l(duration) - initial_conditions[4,1],
                            d_x1(0.0) - initial_conditions[5,0],
                            d_x1(duration) - initial_conditions[5,1],
                            d_x2(0.0) - initial_conditions[6,0],
                            d_x2(duration) - initial_conditions[6,1],
                            d_gamma(0.0) - initial_conditions[7,0],
                            d_gamma(duration) - initial_conditions[7,1],
                            d_theta(0.0) - initial_conditions[8,0],
                            d_theta(duration) - initial_conditions[8,1],
                            d_l(0.0) - initial_conditions[9,0],
                            d_l(duration) - initial_conditions[9,1])

    #Enter the equations of motion in the format f(q,u,t) = 0. A literal copy and paste of the matlab equations F1
    #then find and replace _var with (t), sin with sym.sin, ^2 with **2, dd_x_var with d_x_var.diff()
    eom = sym.Matrix([
                    x1(t).diff() - d_x1(t),
                    x2(t).diff() - d_x2(t),
                    gamma(t).diff() - d_gamma(t),
                    theta(t).diff() - d_theta(t),
                    l(t).diff() - d_l(t),
                    sym.sin(gamma(t))*U(t) - (m_h*(4*d_l(t)*sym.sin(gamma(t) - theta(t))*(d_gamma(t) - d_theta(t)) - 2*d_l(t).diff()*sym.cos(gamma(t) - theta(t)) - 2*d_x1(t).diff() + 2*l(t)*sym.sin(gamma(t) - theta(t))*(d_gamma(t).diff() - d_theta(t).diff()) + 2*l(t)*sym.cos(gamma(t) - theta(t))*(d_gamma(t) - d_theta(t))**2))/2 + d_x1(t).diff()*m_q,
                    (m_h*(2*d_x2(t).diff() + 2*d_l(t).diff()*sym.sin(gamma(t) - theta(t)) + 4*d_l(t)*sym.cos(gamma(t) - theta(t))*(d_gamma(t) - d_theta(t)) + 2*l(t)*sym.cos(gamma(t) - theta(t))*(d_gamma(t).diff() - d_theta(t).diff()) - 2*l(t)*sym.sin(gamma(t) - theta(t))*(d_gamma(t) - d_theta(t))**2))/2 - sym.cos(gamma(t))*U(t) + d_x2(t).diff()*m_q + g*m_h + g*m_q,
                    I*d_gamma(t).diff() - M(t) + d_gamma(t).diff()*l(t)**2*m_h - d_theta(t).diff()*l(t)**2*m_h + 2*d_gamma(t)*d_l(t)*l(t)*m_h - 2*d_l(t)*d_theta(t)*l(t)*m_h + d_x2(t).diff()*l(t)*m_h*sym.cos(gamma(t) - theta(t)) + g*l(t)*m_h*sym.cos(gamma(t) - theta(t)) - d_x1(t).diff()*l(t)*m_h*sym.sin(gamma(t) - theta(t)),
                    -l(t)*m_h*(2*d_gamma(t)*d_l(t) - 2*d_l(t)*d_theta(t) + d_gamma(t).diff()*l(t) - d_theta(t).diff()*l(t) + d_x2(t).diff()*sym.cos(gamma(t) - theta(t)) + g*sym.cos(gamma(t) - theta(t)) - d_x1(t).diff()*sym.sin(gamma(t) - theta(t))),
                    d_l(t).diff()*m_h - L*k + k*l(t) + d_x1(t).diff()*m_h*sym.cos(gamma(t) - theta(t)) + d_x2(t).diff()*m_h*sym.sin(gamma(t) - theta(t)) + g*m_h*sym.sin(gamma(t) - theta(t)) - d_gamma(t)**2*l(t)*m_h - d_theta(t)**2*l(t)*m_h + 2*d_gamma(t)*d_theta(t)*l(t)*m_h
                    ])

    # Specify the objective function and its gradient.
    def obj(free):
        """Minimize the sum of the squares of the control torque."""
        U = free[2 * num_nodes:]
        return interval_value * np.sum(U**2)

    def obj_grad(free):
        grad = np.zeros_like(free)
        grad[2 * num_nodes:] = 2.0 * interval_value * free[2 * num_nodes:]
        return grad

    # Create an optimization problem.
    prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
                known_parameter_map=par_map,
                instance_constraints=instance_constraints,
                bounds=bounds,
                parallel='true')

    #Linearly interploate between the start and end conditions for each state
    initial_guess = np.zeros(0)
    for i in range(len(state_symbols)):
        initial_guess = np.concatenate((initial_guess,np.linspace(initial_conditions[i,0],initial_conditions[i,1],num_nodes)),axis=None)
    #Set the initial guess for each control to 0
    initial_guess = np.concatenate((initial_guess,np.zeros(prob.num_free - len(initial_guess))), axis=None)

    prob.addOption("max_iter",1000)
    prob.addOption("max_cpu_time", 60.0)

    print("Time to formulate problem (s): " + str(time.time() - start))

    solution, info = prob.solve(initial_guess)
    [states_out, specified_values_out, constant_values_out] = parse_free(solution, len(state_symbols), len(specified_symbols),num_nodes)
    
    return np.transpose(states_out)
