import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.stats import linregress



def get_coefficients(H_i,H_w,xi,eta,physics,dimensional=0):
    
    # decide which physics to include
    flexure, water_waves, inertia, buoyancy = physics
    
    # define material properties, elastic moduli, and viscosity (calculated assuming eta = 6 bar*year)
    rho_i = 910
    rho_w = 1000
    g = 9.8
    E = 8.7e9
    nu = 0.3
    mu = E/2/(1+nu)
    t_m = eta/mu
    nu_1 = 3*(1-nu)/(1+nu)
    nu_2 = 3/(2*(1+nu))
    
    if dimensional:
        # inertia
        if inertia:
            N = np.sqrt(rho_w*g/(rho_i*H_i))
            I = 1/(N**2)
        else:
            I = 0

        # water waves
        if water_waves:
            #gamma = 1/np.tanh(H_w*xi)/xi
            gamma = 1/(H_w*xi**2)
            W = gamma/g
        else: 
            W = 0
    else:
        # inertia
        if inertia:
            N = np.sqrt(rho_w*g/(rho_i*H_i))
            I = 1/(t_m**2 * N**2)
        else:
            I = 0

        # water waves
        if water_waves:
            #gamma = 1/np.tanh(H_w*xi)/xi/(t_m**2)
            gamma = 1/(H_w*xi**2)/(t_m**2)
            W = gamma/g
        else: 
            W = 0
        
    # flexure
    if flexure:
        F = (1/3 * H_i**3 * mu * xi**4)/(rho_w*g)
    else:
        F = 0
        
    # buoyancy
    if buoyancy:
        B = 1
    else:
        B = 0
        
    # write the coefficients for the polynomial equation- index corresponds to the degree of each monomial term with respect to s 
    c_4 = (I+W) * nu_1
    c_3 = (I+W) * (nu_1+1)
    c_2 = (I+W) + B*nu_1 + F*nu_2
    c_1 = (B*(nu_1+1) + F)
    c_0 = B
    coefficients = [c_4,c_3,c_2,c_1,c_0]
    
    return coefficients



def asymptotic_solution(xi,H_i,H_w,eta):
    E = 8.7e9
    nu = 0.3
    g = 9.8
    rho_w = 1000
    mu = E/2/(1+nu)
    t_m = eta / mu
    #w = 1/np.tanh(H_w*xi)/xi/(t_m**2)/g
    w = 1/(H_w*xi**2)/(t_m**2)/g
    F = (1/3 * H_i**3 * mu * xi**4)/(rho_w*g)
    s1 = -1 + F*(1/4 - w/4)
    s2 = -2/3 + F*(1/3-4/27*w)
    s3 = -1 * cmath.sqrt(-1/w) + F*(-7/24 - (3*cmath.sqrt(-1/w))/8 + 43*w/216 + 17/72*w*cmath.sqrt(-1/w))
    #s4 = cmath.sqrt(-1/w) + F*(-7/24 + (3*cmath.sqrt(-1/w))/8 + 43*w/216 - 17/72*w*cmath.sqrt(-1/w))
    s4 = cmath.sqrt(-1/w) + F*(-7/24)
    return [s1,s2,s3,s4]


    
def decay_comparison_plot(xi_vect,roots,approximate_solution,T_obs,Q_obs,eta,E,period=np.nan):
    
    # elastic parameters
    nu = 0.3
    mu = E/2/(1+nu)
    t_m = eta/mu
    
    # plot each root 
    fig,ax = plt.subplots(figsize=(15,10))
    
    # plot quality factor Q
    Q = np.abs(roots[:,0].imag/roots[:,0].real/2)
    if np.max(roots[:,0].imag) < 0:
        ax.plot(-1*roots[:,0].imag,Q,label="Numerical")
        ax.set_xlabel('Nondimensional Frequency ($st_m$)')
        #ax.plot(xi_vect,np.abs(roots[0,:].imag/roots[j,0,:].real/2),label="Q")
        #ax.set_xlabel('Wavenumner ($xi$)')
        ax.set_ylabel('Quality factor Q')
    else:
        ax.plot(roots[:,0].imag,Q,label="Numerical")
        ax.set_xlabel('Nondimensional Frequency ($st_m$)')
        #ax.plot(xi_vect,np.abs(roots[0,:].imag/roots[j,0,:].real/2),label="Q")
        #ax.set_xlabel('Wavenumner ($xi$)')
        ax.set_ylabel('Quality factor Q')

    # plot approximate solution
    approximate_Q = abs(approximate_solution.imag/approximate_solution.real)/2
    ax.plot(approximate_solution.imag,approximate_Q,linestyle='-',label="Asymptotic",c='C1')
    #ax.plot(xi_vect,abs(np.array(approximate_solution).imag/np.array(approximate_solution).real)/2)

    # configure labels
    plt.rcParams.update({'font.size': 15})
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both',color='lightgrey')
    
    # plot line at maxwell time and period of interest
    ax.axvline(x=1,c='grey',linestyle='--')
    #ax.text(0.8,0.5,'$t_m$',fontsize=15,c='k')
    ax.axvline(x=t_m/period,c='grey',linestyle='--')
    #ax.text(t_m/period,0.5,'T = '+str(period)+'s',fontsize=15,c='k')
    ax.set_xticks([1,10,100,1000,10000,100000])
    ax.text(t_m/period+0.25,1000,'T = '+str(period)+'s',fontsize=15,c='k',rotation=90)
    ticklabels = ['$s=\dfrac{1}{t_m}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$']
    ax.set_xticklabels(ticklabels)
    
    # label Q value for desired period
    period_Q = Q[np.argmin(abs(np.abs(roots[:,0].imag)-t_m/period))]
    power = np.floor(np.log10(period_Q))
    num = np.round(period_Q/(10**power),2)
    ax.text(t_m/period,period_Q*2,'  $Q$ = '+str(num)+"e"+str(int(power)),fontsize=15,c="C0")
    
    # label approximate Q value for desired period
    approx_period_Q = approximate_Q[np.argmin(abs(np.abs(approximate_solution.imag)-t_m/period))]
    power = np.floor(np.log10(approx_period_Q))
    num = np.round(approx_period_Q/(10**power),2)
    ax.text(t_m/period,approx_period_Q,'  $Q_{asym}$ = '+str(num)+"e"+str(int(power)),fontsize=15,c='C1')

    power = np.floor(np.log10(eta))
    num = int(eta/(10**power)) if (eta/(10**power)).is_integer() else eta/(10**power)
    
    # plot observations
    ax.scatter(t_m/T_obs,Q_obs,color='k',label="Observations\n(PIG2 HHZ)")
    reg = linregress(np.log10(t_m/T_obs),np.log10(Q_obs))
    reg_line = 10**(reg.intercept + reg.slope*np.log10(t_m/T_obs))
    plt.plot(t_m/T_obs,reg_line,linestyle='--',color='C2',label="Best fit")
    Q_fit = 10**(reg.intercept + reg.slope*np.log10(t_m/period))
    ax.text(t_m/period,Q_fit,'  $Q_{fit}$ = '+ str(np.round(Q_fit,2)),fontsize=15,c="C2")
    ax.legend()

    # set limits
    ax.set_ylim(1,10**6.5)
    ax.set_xlim(0.5,10**5)
    
    # make title
    power = np.floor(np.log10(eta))
    num = int(eta/(10**power)) if (eta/(10**power)).is_integer() else eta/(10**power)
    eta_string = "$\eta=" + str(num)+"e"+str(int(power))+"$"
    power = np.floor(np.log10(E))
    num = int(E/(10**power)) if (E/(10**power)).is_integer() else E/(10**power)
    youngs_string = "$E=" + str(num)+"e"+str(int(power))+"$"
    ax.set_title("Floating viscoelastic beam with viscosity "+eta_string+" and Young's modulus "+youngs_string,size=15)
    return