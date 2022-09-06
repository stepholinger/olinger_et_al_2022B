import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fmin
import matplotlib.pyplot as plt



def stress_intensity(L,sigma):
    return sigma*np.sqrt(np.pi*L)



def find_min_rxx(L0,sigma0,K_ic):
    # Find Rxx such that K_i = K_ic assuming no dynamic contribution from water level
    goal = lambda R : np.abs(K_ic - stress_intensity(L0,R+sigma0))
    Rxx_best = fmin(goal, 10e3)
    return Rxx_best



def model_fracture(y, t, epsilon, H_i, Rxx, w0):
    # get variables and set parameters
    eta, d_eta_dt, L = y
    g = 9.8
    v_r = 2000
    K_c = 1e5
    rho_i = 910
    rho_w = 1000
    H_w = rho_i/rho_w * H_i


    # calculate stress and stress intensity factor
    sigma = rho_i*g*H_i/2 - rho_w*g/(2*H_i)*eta**2
    sigma_sum = Rxx - sigma
    K_i = stress_intensity(L,sigma_sum)
    
    # use stress intensity factor to determine if crack propagates
    if K_i < K_c:
        dLdt = 0
    else:
        dLdt = v_r * (1 - (K_c/K_i)**2)
        
    # use "Eshelby" relation to get width and its time derivative as a function of length
    mu_star = 3.6e9/(1-0.33)
    alpha = sigma_sum / mu_star * np.pi/4
    w = w0 + alpha * L

    # write time derivative of stress sum
    d_sigma_sum_dt = d_eta_dt*eta*rho_w*g/H_i
    
    # write time derivative of alpha
    d_alpha_dt = d_sigma_sum_dt*np.pi/mu_star/4
    
    # write time derivative of width
    dwdt = alpha*dLdt + d_alpha_dt*L
        
    # write time derivative of water volume, V'
    dVdt = w*L*d_eta_dt + w*dLdt*eta + dwdt*L*eta

    # write 2nd time derivative of crack length
    dLd2t = (v_r*K_c**2)/np.pi * (1/((L**2)*(sigma_sum**3))*(dLdt*sigma_sum + 2*L*d_sigma_sum_dt))
    
    # write expression for terms of V" that do not contain eta"
    gamma = rho_w*g*np.pi/4/H_i/mu_star
    lamb = 2*(w*dLdt*d_eta_dt + dwdt*L*d_eta_dt + dwdt*dLdt*eta) + w*dLd2t*eta + L*eta*(alpha*dLd2t + 2*d_alpha_dt*dLdt + gamma*L*d_eta_dt**2)
    beta = L*(w + gamma*L*eta**2)
    
    # write bernoulli with u' substituted in so all eta" terms can be moved to LHS
    d_eta_d2t = (H_w + epsilon*beta/L)**-1 * (g*(H_w - eta) - (epsilon/L**2)*(lamb*L - dVdt*dLdt))
    
    dydt = [d_eta_dt, d_eta_d2t, dLdt]
    
    return dydt