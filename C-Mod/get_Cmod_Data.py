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
def currentShot(conn):
    return conn.get('current_shot("cmod")').data()

###############################################################################
class BP:
    # Low frequency M/N Mirnov Array
    
    def __init__(self,shotno,debug=False):
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        basePath= r'\CMOD::TOP.MHD.MAGNETICS:BP_COILS:'
        self.BC = {'NAMES':[],'PHI':[],'THETA':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        self.DE = {'NAMES':[],'PHI':[],'THETA':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        self.GH = {'NAMES':[],'PHI':[],'THETA':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        self.JK = {'NAMES':[],'PHI':[],'THETA':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        
        self.nodeName = conn.get(basePath + 'NODENAME').data()
        phi = conn.get(basePath + 'PHI').data()
        theta = conn.get(basePath + 'THETA_POL').data()
        theta_tor = conn.get(basePath + 'THETA_TOR').data()
        R = conn.get(basePath + 'R').data()
        Z = conn.get(basePath + 'Z').data()
        
        for ind,name in enumerate(self.nodeName):
                signal = conn.get(basePath+'SIGNALS:%s'%name)
                
                # Reference pass link node
                if name[-2::] == 'BC': tmp = self.BC
                if name[-2::] == 'DE': tmp = self.DE
                if name[-2::] == 'GH': tmp = self.GH
                if name[-2::] == 'JK': tmp = self.JK
                
                tmp['NAMES'].append(str(name))
                tmp['PHI'].append(float(phi[ind]))
                tmp['THETA'].append(float(theta[ind]))
                tmp['THETA_TOR'].append(float(theta_tor[ind]))
                tmp['R'].append(float(R[ind]))
                tmp['Z'].append(float(Z[ind]))
                
                tmp['SIGNAL'].append(signal.data())
                
                
        self.time = conn.get('dim_of('+basePath+'SIGNALS:%s)'%name).data()
        


###############################################################################
class BP_T:
    # High frequency Mirnov array
    
    def __init__(self,shotno,debug=False):
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
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
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
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
class FRCECE:
    # High frequency ECE (FR-C-ECE-F), not availible for shots ~< 2010
    
    def __init__(self,shotno):
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        basePath = '\CMOD::TOP.ELECTRONS:FRCECE:DATA:'
        self.ECE = []
        self.R = []
        for i in np.arange(1,33):
            self.ECE.append(conn.get(basePath+'ECEF%02d'%i).data())
        self.R.append(conn.get(basePath+'R').data())
        self.ECE = np.array(self.ECE); self.R=np.array(self.R).squeeze()
        self.time = conn.get('dim_of('+basePath+'ECEF%02d'%i+')').data()
        # separate timebase for EFIT reconstructions
        self.time_R = conn.get('dim_of('+basePath+'R'+')').data()
    
    def makePlot(self,ax=None,downsample=100,figParams={}):
        self.R=self.R.squeeze()
        if ax is None:
            plt.close('C-ECE-Profile')
            fig,ax=plt.subplots(1,1,num='C-ECE-Profile',tight_layout=True,figsize=(6,3))
        else: fig = plt.gcf()
        ax.contourf(self.time[::downsample],np.mean(self.R,axis=1),
                    self.ECE[:,::downsample],zorder=-3,cmap='plasma',
                    levels=np.linspace(-1,np.max(self.ECE)*.5))
        ax.set_rasterization_zorder(-1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('R [m]')
        norm = Normalize(-1,np.max(self.ECE)*.5)
        if 'noBar' not in figParams: 
            fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax,
                     label=r'CECE [V?]')
        plt.show()
###############################################################################
class Ip:
    # Plasma current
    
    def __init__(self,shotno):
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        self.ip = -1*conn.get('\MAGNETICS::IP').data()
        self.time = conn.get('dim_of(\MAGNETICS::IP)').data()
        conn.closeAllTrees()
        
    def makePlot(self,ax=None):
        if ax is None:
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
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        self.pwr = conn.get(r'\CMOD::TOP.RF.ANTENNA:RESULTS:PWR_NET_TOT').data()
        self.time = conn.get(r'dim_of(\CMOD::TOP.RF.ANTENNA:RESULTS:PWR_NET_TOT)').data()
        self.shotno=shotno
        conn.closeAllTrees()
        
    def makePlot(self,ax=None,figParams={}):
        if ax is None:
            plt.close('RF_Pwr')
            fig,ax=plt.subplots(1,1,num='RF_Pwr',figsize=(6,3),tight_layout=True)
        ax.plot(self.time,self.pwr,label=self.shotno,\
                c=figParams['c'] if 'c' in figParams else plt.get_cmap('tab10')(0),\
                    alpha=figParams['alpha'] if 'alpha' in figParams else 1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\mathrm{P_{rf}}$ [MW]')
        if 'noGrid' not in figParams: ax.grid()
        if 'noLeg' not in figParams: ax.legend(fontsize=8,handlelength=1)
        plt.show()
        
###############################################################################
        