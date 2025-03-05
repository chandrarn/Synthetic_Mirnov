# -*- coding: utf-8 -*-
"""
Spyder Editor
    Gen C-Mod data for a variety of diagnostics
"""


from header_Cmod import np, plt, mds, Normalize, cm

###############################################################################
def openTree(shotno):
    # Connect to data tree
    conn = mds.Connection('alcdata')
    conn.openTree('CMOD',shotno)
    return conn

###############################################################################
class BP_T:
    # High frequency Mirnov array
    
    def __init__(self,shotno,debug=False):
        conn = openTree(shotno)
        
        basePath= r'\CMOD::TOP.MHD.MAGNETICS:ACTIVE_MHD:SIGNALS:'
        self.ab_data = []   
        self.ab_names = []
        self.ab_theta = []
        self.gh_data = []
        self.gh_names = []
        self.gh_theta = []
        
        for i in np.arange(1,7):
            
            try:# not ever node is on on every shot
                node = 'BP%dT_ABK'%i
                self.ab_data.append(conn.get(basePath+node).data())
                self.ab_names.append(node)
                if debug:print('Loaded %s'%node)
            except:pass
            
            try:
                node = 'BP%dT_GHK'%i
                self.gh_data.append(conn.get(basePath+node).data())
                self.gh_names.append(node)
                if debug:print('Loaded %s'%node)
            except:pass
            
        self.ab_data = np.array(self.ab_data)
        self.gh_data = np.array(self.gh_data)
        
        # Assumption: timebase is the same for all nodes, we just need to grab 
        # the first one that works, agnostic to which specific node it is
        t_load_success = 0;i=1;
        while not t_load_success:    
            try:
                self.time = conn.get('dim_of('+basePath+'BP%dT_ABK'%i+')').data()
                t_load_success=1
                if debug:print('Loaded Timebase ' +  'BP%dT_ABK'%i)
            except:i+=1
        
        conn.closeAllTrees()
    
    def makePlot(self):
        plt.close('BP_T')
        fig,ax=plt.subplots(2,3,num='BP_T',tight_layout=True,sharex=True,sharey=True)
        
        for i in range(6):
            try: ax[np.unravel_index(i, (2,3))].plot(self.time,self.ab_data[i],\
                            c=plt.get_cmap('tab10')(0),label=self.ab_names[i])
            except:pass
            try: ax[np.unravel_index(i, (2,3))].plot(self.time,self.gh_data[i],
                        alpha=.5,c=plt.get_cmap('tab10')(1),label=self.gh_names[i])
            except: pass
            ax[np.unravel_index(i, (2,3))].grid()
            ax[np.unravel_index(i, (2,3))].legend(fontsize=8,handlelength=1,loc='upper left')
        for i in range(2):ax[i,0].set_ylabel('Signal [arb?]')
        for i in range(3):ax[1,i].set_xlabel('Time [s]')
        
        plt.show()
        
    
###############################################################################
class ECE:
    # ECE radial profile [not high frequency ECE data]
    
    def __init__(self,shotno,debug=True):
        conn = openTree(shotno)
        self.Te=conn.get('\CMOD::TOP.ELECTRONS:ECE:RESULTS:ECE_TE').data()
        self.R=conn.get('dim_of(\CMOD::TOP.ELECTRONS:ECE:RESULTS:ECE_TE,0)').data()
        self.time=conn.get('dim_of(\CMOD::TOP.ELECTRONS:ECE:RESULTS:ECE_TE,1)').data()
        conn.closeAllTrees()
    
    def makePlot(self):
        plt.close('ECE-Profile')
        fig,ax=plt.subplots(1,1,num='ECE-Profile',tight_layout=True,figsize=(6,3))
        ax.contourf(self.time,self.R,self.Te.T,levels=50,zorder=-3,cmap='plasma')
        ax.set_rasterization_zorder(-1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('R [m]')
        norm = Normalize(np.min(self.Te)*1e-3,np.max(self.Te)*1e-3)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax,
                     label=r'$\mathrm{T_e}$ [keV]')
        plt.show()

###############################################################################
class Ip:
    # Plasma current
    
    def __init__(self,shotno):
        conn = openTree(shotno)
        self.shotno=shotno
        self.ip = -1*conn.get('\MAGNETICS::IP').data()
        self.time = conn.get('dim_of(\MAGNETICS::IP)').data()
        conn.closeAllTrees()
        
    def makePlot(self):
        plt.close('Ip')
        fig,ax=plt.subplots(1,1,num='Ip',tight_layout=True,figsize=(6,3))
        ax.plot(self.time,self.ip*1e-3,label=self.shotno)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\mathrm{I_p}$ [kA]')
        ax.grid()
        ax.legend(fontsize=8)
        plt.show()
        
###############################################################################
class RF_PWR():
    # ICRF Injected power
    
    def __init__(self,shotno):
        conn = openTree(shotno)
        self.pwr = conn.get(r'\CMOD::TOP.RF.ANTENNA:RESULTS:PWR_NET_TOT').data()
        self.time = conn.get(r'dim_of(\CMOD::TOP.RF.ANTENNA:RESULTS:PWR_NET_TOT)').data()
        self.shotno=shotno
        conn.closeAllTrees()
        
    def makePlot(self):
        plt.close('RF_Pwr')
        fig,ax=plt.subplots(1,1,num='RF_Pwr',figsize=(6,3),tight_layout=True)
        ax.plot(self.time,self.pwr,label=self.shotno)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\mathrm{P_{rf}}$ [MW]')
        ax.grid()
        ax.legend(fontsize=8,handlelength=1)
        plt.show()
        
###############################################################################
        