import numpy as np 
import matplotlib.pyplot as plt 
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.25, style='white')

### interactive display
from IPython.display import display,clear_output


class eGrid(object):
    
    def __init__(self, phase_diff=0., grid_option = 'flat'):
        self.E1 = np.array([0., 5])
        self.E2 = np.array([10., 5])
        self.ER = np.array([5, 0.])
    
        self.N = 11
        self.Recording_x = np.arange(self.N)
        
        if grid_option == 'slanted':
            self.Recording_y = 0.25*self.Recording_x + 1.5
        
        elif grid_option == 'superslanted':
            self.Recording_y = 0.5*self.Recording_x - 1
            
        elif grid_option == 'flat': 
            self.Recording_y = np.zeros((self.N)) + 4

        self.grid_option = grid_option
    
        ### Reference electrode ###
        self.RR = np.array([5, 10.])
        self.V = np.zeros((self.N,))
        
        ### get distances from E1/E2; 
        self.dist_E1 = ((self.Recording_x - self.E1[0])**2 + (self.Recording_y - self.E1[1])**2)**.5
        self.dist_E2 = ((self.Recording_x - self.E2[0])**2 + (self.Recording_y - self.E2[1])**2)**.5
        
        ### Get distances from E1/E2 to the reference electrode ###
        self.dist_E1_R = (np.sum((self.RR - self.E1)**2)**.5)
        self.dist_E2_R = (np.sum((self.RR - self.E2)**2)**.5)
        
        ### calculations 
        self.k = 9*10**9 #(Nm**2/c**2) ### V = k*q/r, k is a constant 
        
        ## integration time for stimulation wavefrom (basically to get 
        ## q for the above expression, integrate the waveform within time 
        ## step of dt ( q = amps*dt )
        self.dt = 1*10**-6 # ms;  
        
        ### How many tiem steps are in a 100 ms cycle ? 
        self.t_in_100ms = int(.1/self.dt)
        
        ## Phase differnece between E1 and E2 in radians 
        self.phase_diff = phase_diff
        
        ### colors for plotting 
        self.colors = ['maroon', 'orangered', 'deeppink','darkgoldenrod', 'olivedrab',
                       'teal', 'steelblue', 'midnightblue', 'darkmagenta', 'black', 'gray']

    def plot(self):
        ### Plot the electorde strip ##
        f, ax = plt.subplots(figsize = (10, 10))
        ax.set_title('Electrode Config %s'%(self.grid_option))
        ax.plot(self.E1[0], self.E1[1], 'rs', markersize=20)
        ax.plot(self.E2[0], self.E2[1], 'rs', markersize=20)
        ax.plot(self.ER[0], self.ER[1], 'b*', markersize=40)
        for i in range(self.N):
            ax.plot(self.Recording_x[i], self.Recording_y[i], '.', color=self.colors[i],
                   markersize=20)
        ax.plot(self.RR[0], self.RR[1], 'k*', markersize=40)

        ### text labels: 
        ax.text(self.RR[0], self.RR[1]+0.5, 'Recording ref', ha='center')
        ax.text(self.E1[0], self.E1[1]+0.75, 'Stim 1', ha='center',color='r')
        ax.text(self.E2[0], self.E2[1]+0.75, 'Stim 2', ha='center',color='r')
        ax.text(self.ER[0], self.ER[1]+0.5, 'Stim Return', ha='center',color='b')
        
        ### Contact labels 
        ax.text(self.Recording_x[0], self.Recording_y[0]-0.5, 'Contact 1', 
            ha='center',color=self.colors[0])
        ax.text(self.Recording_x[-1], self.Recording_y[-1]-0.5, 'Contact %d'%self.N, 
            ha='center',color=self.colors[self.N-1])
        
        ax.set_ylim([-1., 11])
        ax.set_xlim([-1., 11])
        
    def _compute_v(self, t):
        ### Get the voltage at each point in space due to E1 / E2
        ### Then compute the voltage differences (pt in space minus reference electrode)
        
        ### Get amps 
        a1, a2 = self._compute_a(self.phase_diff, t)
        
        ## Get charage on each electrode at this point in the stim wf
        q_e1, q_e2 = self._compute_q(a1, a2)
        
        #### We know how much charge there is at each electrode; 
        ### Compute voltage at each point due to this; 
        
        ### Get voltage at the reference
        v_r_1 = (self.k*q_e1/self.dist_E1_R) + (self.k*q_e2/self.dist_E2_R)
        
        ### Get teh electrode voltages 
        v_e = (self.k*q_e1/self.dist_E1) + (self.k*q_e2/self.dist_E2)
        
        ### Get voltage w.r.t. reference ###
        self.V = v_e - v_r_1; 
    
    def _compute_q(self, a1=0, a2=0): 
        ### Convert amps to q; 
        return a1*self.dt, a2*self.dt 
    
    def _compute_a(self, phase_diff, t): 
        ### 1mA ###
        a1 = 1*10**-3*np.sin(2*np.pi*10*t)
        a2 = 1*10**-3*np.sin((2*np.pi*10*t) + phase_diff)
        return a1, a2
    
    def plot_volts(self, skip_plot=False):
        
        ### Plot voltages as a function of time on each electrode 
        ts = np.linspace(0, .1, self.t_in_100ms)
        ts = ts[::1000]
        V = []
        for i in range(len(ts)):
            self._compute_v(ts[i])
            V.append(self.V.copy())
        V = np.vstack((V))
        
        ### Get max voltages for the max voltage plot 
        self.maxV = np.max(V, axis=0)
        
        if skip_plot:
            pass
        else:
            f, ax = plt.subplots()
            for i in range(self.N):
                ax.plot(ts+.0001*i, V[:, i], color=self.colors[i])
            ax.set_title('Phase offs = %d deg'%int(self.phase_diff/np.pi*180))
            ax.set_ylabel('Voltage')
            ax.set_xlabel('Time (sec)')

def opitz_fig2(grid_option): 
    X = np.zeros((9, 11))
    
    ### Iterate through the angles ###
    for i_a, ang in enumerate(np.linspace(0, 360, 9)):
        x = eGrid(phase_diff=float(ang)/180.*np.pi, grid_option=grid_option)

        if ang == 0: 
            x.plot()

        ### Do the voltage plots ###
        if ang in [0, 90, 180, 270]:
            x.plot_volts(skip_plot = False)
        else:
            x.plot_volts(skip_plot = True)
        X[i_a, :] = x.maxV.copy()

        if ang == 180:
            argmax = np.argmax(x.maxV)
            print('Argmax 180 %d' %argmax)
    
    f, ax = plt.subplots(figsize =(5, 8))
    ax.set_title('Electrode config: %s' %(grid_option))
    cax = ax.pcolormesh(np.linspace(0, 360, 9), np.arange(X.shape[1]+1), np.flipud(X.T), cmap='viridis')
    ax.set_yticks(np.arange(1, 12) - 0.5)
    ax.set_yticklabels(np.arange(1, 12)[::-1])
    ax.set_xlabel('Phase Diff (degrees)')
    ax.set_ylabel('Contacts')
    cb=f.colorbar(cax, ax=ax)
    cb.set_label('Voltage',rotation=270)
    f.tight_layout()
        
