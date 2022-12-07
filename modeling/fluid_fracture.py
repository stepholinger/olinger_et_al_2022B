import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from sklearn import linear_model


def stress_intensity(L,sigma):
    return sigma*np.sqrt(np.pi*L)



def find_min_rxx(L0,sigma0,K_ic):
    # Find Rxx such that K_i = K_ic assuming no dynamic contribution from water level
    goal = lambda R : np.abs(K_ic - stress_intensity(L0,R+sigma0))
    Rxx_best = fmin(goal, 10e3)
    return Rxx_best



def model_fracture(y, t, epsilon, H_i, v_r, Rxx, w0):
    # get variables and set parameters
    eta, d_eta_dt, L = y
    g = 9.8
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



def model_fracture_no_coupling(y, t, epsilon, H_i, Rxx, w0):
    # get variables and set parameters
    L = y
    g = 9.8
    v_r = 2000
    K_c = 1e5
    rho_i = 910
    rho_w = 1000
    H_w = rho_i/rho_w * H_i

    # calculate stress and stress intensity factor
    sigma_sum = Rxx - (rho_i*g*H_i/2 - rho_w*g/(2*H_i)*H_w**2)
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
    
    dydt = dLdt
    
    return dydt


def model_fracture_stopping(y, t, epsilon, L1, H_i, Rxx, w0):
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
    if L < L1:
        if K_i < K_c:
            dLdt = 0
        else:
            dLdt = v_r * (1 - (K_c/K_i)**2)
    if L > L1:
        dLdt = 0
        
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
    if L > L1:
        dLd2t = 0
        
    # write expression for terms of V" that do not contain eta"
    gamma = rho_w*g*np.pi/4/H_i/mu_star
    lamb = 2*(w*dLdt*d_eta_dt + dwdt*L*d_eta_dt + dwdt*dLdt*eta) + w*dLd2t*eta + L*eta*(alpha*dLd2t + 2*d_alpha_dt*dLdt + gamma*L*d_eta_dt**2)
    beta = L*(w + gamma*L*eta**2)
    
    # write bernoulli with u' substituted in so all eta" terms can be moved to LHS
    d_eta_d2t = (H_w + epsilon*beta/L)**-1 * (g*(H_w - eta) - (epsilon/L**2)*(lamb*L - dVdt*dLdt))
    
    dydt = [d_eta_dt, d_eta_d2t, dLdt]
    
    return dydt



def plot_fractures(t,sol,sol_no_coupling,c_r,H_w,L0,L1,c_avg,ylims):
    # get some useful variables
    dLdt = np.gradient(sol[:, 2],t)
    dLdt_no_coupling = np.gradient(sol_no_coupling[:, 0],t)

    # plot comparison of solutions with and without coupling
    fig,ax = plt.subplots(4,1,figsize=(8,11),dpi=150)
    fig.patch.set_facecolor('w')
    ax[0].hlines(t[0],t[-1],0,label='Without fluid coupling',colors='darkorange')
    ax[0].plot(t, sol[:, 0]-H_w,label='With fluid coupling',c='k')
    ax[0].set_ylabel('Deflection $\eta$ (m)')
    ax[0].legend()
    ax[0].set_title("Deflection $\eta$ from hydrostatic water level through time")
    ax[1].plot(t, dLdt_no_coupling/c_r,label='Without fluid coupling',c='darkorange')
    ax_coupling = ax[1].twinx()
    ax_coupling.plot(t, dLdt/c_r,label='With fluid coupling',c='k')
    ax[1].set_ylabel('$dLdt\;/\;c_r$',c='k')
    ax_coupling.set_ylabel('$dLdt\;/\;c_r$',c='k')
    ax[1].set_title("Propagation rate $dLdt$ through time")
    ax[2].plot(t, sol_no_coupling[:, 0]/1e3, label='Without fluid coupling',c='darkorange')
    ax[2].plot(t, sol[:, 2]/1e3, label='With fluid coupling',c='k')
    ax[2].set_ylabel('Length $L$ (km)')
    ax[2].set_title("Fracture length $L$ through time")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_ylim(ylims[0])
    ax[1].set_ylim(ylims[1])
    ax[2].set_ylim(ylims[2])
    ax_coupling.set_ylim([0,np.max(dLdt/c_r)*1.5])
    ax[3].set_ylim(ylims[3])
    ax[0].set_xlim([0,t[-1]])
    ax[1].set_xlim([0,t[-1]])
    ax[2].set_xlim([0,t[-1]])
    
    
    # subset before length
    L = sol[:, 2]
    idx = np.where(L < L1)[0]
    sol = sol[idx,:]
    sol_no_coupling = sol_no_coupling[idx,:]
    t_zoom = t[idx]
    
    # plot zoom between L0 and L1
    ax[3].plot(t_zoom, sol_no_coupling[:, 0]/1e3, label='Without fluid coupling',c='darkorange')
    ax[3].plot(t_zoom, sol[:, 2]/1e3, label='With fluid coupling',c='k')
    ax[3].set_ylabel('Length $L$ (km)')
    ax[3].grid()
    ax[3].set_xlim([0,t_zoom[-1]])
    ax[3].set_title("Fracture length $L$ through time over " + str(np.round(t_zoom[-1],0)).split(".")[0] + " s")
    ax[3].set_xlabel('Time (seconds)')
    
    # add average speed line
    ax[3].plot(t_zoom,L0/1000+t_zoom*c_avg/1000,linestyle='--',dashes=(4,4),color='purple')
    theta = np.arctan((L1-L0)/1000/t_zoom[-1])*180/np.pi
    trans_angle = ax[3].transData.transform_angles(np.array((theta,)),np.array((100, 8)).reshape((1, 2)))[0]
    ax[3].text(100,8,"$c_{avg}$ = " + str(c_avg) + ' m/s',c='purple',rotation=trans_angle)
    
    # highlight L0 to L1
    rect = Rectangle((0, ylims[2][0]), t_zoom[-1]-t_zoom[0], ylims[2][1]-ylims[2][0], linewidth=0, facecolor='r',alpha=0.05,zorder=0)
    ax[2].axvline(t_zoom[-1],linewidth=0.75,linestyle='--',dashes=(4,4),color='k')
    ax[2].add_patch(rect)
    
    # draw lines connecting the highlighted area of first data plot to second plot
    line1_start = [0,ax[3].get_ylim()[1]]
    line1_end = [0,ax[2].get_ylim()[0]]
    line2_start = [t_zoom[-1],ax[3].get_ylim()[1]]
    line2_end = [t_zoom[-1],ax[2].get_ylim()[0]] 
    con1 = ConnectionPatch(xyA=line1_start, xyB=line1_end, coordsA='data', coordsB='data',axesA=ax[3],axesB=ax[2],linestyle=(5,(4,4)))
    con2 = ConnectionPatch(xyA=line2_start, xyB=line2_end, coordsA='data', coordsB='data',axesA=ax[3],axesB=ax[2],linestyle=(5,(4,4)))
    ax[2].add_artist(con1)
    ax[3].add_artist(con2)
    cons = [con1,con2]
    for con in cons:
        con.set_color('k')
        con.set_linewidth(0.75)
    
    # display plot
    plt.tight_layout()
    plt.savefig("outputs/figures/fracture_model.png",bbox_inches="tight")