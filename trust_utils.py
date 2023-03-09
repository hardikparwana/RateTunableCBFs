import numpy as np

def compute_trust(A,b,xdotj,xdotj_nominal,h,min_dist,h_min): # h > 0
    
    # distance
    rho_dist = A @ xdotj - b;
    # try: 
    #     assert(rho_dist>0.001) # should always be positive
    # except:
    #     print(f"Assertion failed rho_dist:{rho_dist}") 

    # if h < h_min:
    #     print(f"small h: {h}")
    rho_dist = np.tanh(0.1*rho_dist) #np.tanh(3.0*rho_dist); # score between 0 and 1  
    
    # angle
    if np.linalg.norm(xdotj)>0.01:
        theta_as = np.real(np.arccos( A @ xdotj/np.linalg.norm(A)/np.linalg.norm(xdotj) / 1.05))
    else:
        theta_as = np.arccos(0.001)
    if np.linalg.norm(xdotj_nominal)>0.01:
        theta_ns = np.real(np.arccos( A @ xdotj_nominal/np.linalg.norm(A)/np.linalg.norm(xdotj_nominal)/1.05 )) 
    else:
        theta_ns = np.arccos(0.001)
    # if (theta_ns<0.05):
    #     theta_ns = 0.05

    rho_theta = np.tanh(theta_ns/theta_as*0.55) # if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
    # print(f"rho_dist:{rho_dist}")
    if rho_dist>min_dist: # always positive
        trust = 2*rho_theta*rho_dist#(rho_dist-min_dist)
    else: # danger
        if h>h_min:  # far away. therefore still relax/positive
            trust = 2*rho_theta*rho_dist
        else:  # definitely negative this time
            # print("Negative Trust!")
            trust = -2*(1-rho_theta)*rho_dist
            
    asserted = True
    try: 
        assert(rho_dist>0.001) # should always be positive
    except:
        print(f"Assertion failed rho_dist:{rho_dist}, trust:{trust}") 
        asserted = False
        
    return trust, asserted