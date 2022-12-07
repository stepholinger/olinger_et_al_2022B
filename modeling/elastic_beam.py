def fg_dispersion_coeffs(h_i,h_w,f):
    rho_i = 910
    rho_w = 1000
    g = 9.8
    E = 8.7e9
    nu = 0.3
    D = (E*h_i**3)/(12*(1-nu**2))
    return [D,0,0,0,rho_w*g-rho_i*h_i*f**2,0,-(rho_w/h_w)*f**2]