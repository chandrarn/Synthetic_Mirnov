# -*- coding: utf-8 -*-
"""
Spyder Editor
    Gen C-Mod data for a variety of diagnostics
"""


from header_Cmod import np, plt, mds, Normalize, cm, gaussianHighPassFilter, \
    gaussianLowPassFilter, __doFilter, pk, json, MDSplus, data_archive_path
from mirnov_Probe_Geometry import Mirnov_Geometry
        
    
###############################################################################
def openTree(shotno,treeName='CMOD'):
    # Connect to data tree
    conn = mds.Connection('alcdata')
    conn.openTree(treeName,shotno)
    return conn

###############################################################################
def currentShot(conn):
    return conn.get('current_shot("cmod")').data()

############################################################################
def doFilter(data,time,HP_Freq,LP_Freq): # Issue with names with '__' in classes
    return __doFilter(data, time, HP_Freq, LP_Freq)
    
###############################################################################
class BP:
    # Low frequency M/N Mirnov Array
    
    def __init__(self,shotno,debug=False):
        if debug:print('Loading Low m/n B-Dot Signals')
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        basePath= r'\CMOD::TOP.MHD.MAGNETICS:BP_COILS:'
        self.BC = {'NAMES':[],'PHI':[],'THETA_POL':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        self.DE = {'NAMES':[],'PHI':[],'THETA_POL':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        self.GH = {'NAMES':[],'PHI':[],'THETA_POL':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        self.JK = {'NAMES':[],'PHI':[],'THETA_POL':[],'THETA_TOR':[],'R':[],'Z':[],'SIGNAL':[]}
        
        self.nodeName = conn.get(basePath + 'NODENAME').data()
        phi = conn.get(basePath + 'PHI').data()
        theta = conn.get(basePath + 'THETA_POL').data() # Note that this is an orientation
        theta_tor = conn.get(basePath + 'THETA_TOR').data() # Note that this is an orientation
        R = conn.get(basePath + 'R').data()
        Z = conn.get(basePath + 'Z').data()
        
        for ind,name in enumerate(self.nodeName):
                try:
                    signal = conn.get(basePath+'SIGNALS:%s'%name)
                    assert type(signal.data()) is np.ndarray # signal node may exist but be empty
                except: continue
                # Reference pass link node
                if name[-2::] == 'BC': tmp = self.BC
                if name[-2::] == 'DE': tmp = self.DE
                if name[-2::] == 'GH': tmp = self.GH
                if name[-2::] == 'JK': tmp = self.JK
                
                tmp['NAMES'].append(str(name))
                tmp['PHI'].append(float(phi[ind]))
                tmp['THETA_POL'].append(float(theta[ind]))
                tmp['THETA_TOR'].append(float(theta_tor[ind]))
                tmp['R'].append(float(R[ind]))
                tmp['Z'].append(float(Z[ind]))
                
                tmp['SIGNAL'].append(signal.data())
                
        # Convert to numpy arrays
        for i in range(4):
            if i == 0: tmp = self.BC
            if i == 1: tmp = self.DE
            if i == 2: tmp = self.GH
            if i == 3: tmp = self.JK
            
            tmp['NAMES'] = np.array(tmp['NAMES'])
            tmp['PHI'] = np.array(tmp['PHI'])
            tmp['THETA_POL'] = np.array(tmp['THETA_POL'])
            tmp['THETA_TOR'] = np.array(tmp['THETA_TOR'])
            tmp['R'] = np.array(tmp['R'])
            tmp['Z'] = np.array(tmp['Z'])
            tmp['SIGNAL'] = np.array(tmp['SIGNAL'])
        self.time = conn.get('dim_of('+basePath+'SIGNALS:%s)'%name).data()
        
    def makePlot(self,HP=10,LP=20e3,tLim=[1,1.1]):
        plt.close('BP_Contour')
        fig,ax=plt.subplots(2,1,tight_layout=True,num='BP_Contour',sharex=True)
        
        out= doFilter(self.BC['SIGNAL'], self.BC.time, HP, LP)
        
        dt = self.time[1]-self.time[0]
        tInds = np.arange(*[(t-self.time[0])/dt for t in tLim],dtype=int)
        
        theta = np.arctan2(self.BC['Z'],self.BC['R']-.67)*180/np.pi
        ax[0].contourf(theta,self.time[tInds],out[:,tInds])
        
        ax[0].set_ylabel(r'$\hat{\theta}$ [deg]')
        ax[1].set_xlabel('Time [s]')
        
        
        plt.show()


###############################################################################
class BP_T:
    # High frequency Mirnov array
    
    def __init__(self,shotno,debug=False):
        if debug: print('Loading High-Frequency Mirnovs (BPT)')
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        basePath= r'\CMOD::TOP.MHD.MAGNETICS:ACTIVE_MHD:SIGNALS:'
        self.ab_data = []   
        self.ab_names = []
        self.ab_theta_pol = []
        self.ab_phi = []
        self.ab_r = []
        self.ab_z = []
        
        self.gh_data = []
        self.gh_names = []
        self.gh_theta_pol = []
        self.gh_phi = []
        self.gh_r = []
        self.gh_z = []
        
        # Presaved spatial geometetry dicts
        phi, theta_pol, R, Z = Mirnov_Geometry(self.shotno)
        
        for i in np.arange(1,7):
            
            try:# not ever node is on on every shot
                node = 'BP%dT_ABK'%i
                self.ab_data.append(conn.get(basePath+node).data())
                self.ab_names.append(node)
                
                self.ab_theta_pol.append(theta_pol[node])
                self.ab_phi.append(phi[node])
                self.ab_r.append(R[node])
                self.ab_z.append(Z[node])
                if debug:print('Loaded %s'%node)
            except:pass
            
            try:
                node = 'BP%dT_GHK'%i
                self.gh_data.append(conn.get(basePath+node).data())
                self.gh_names.append(node)
                
                self.gh_theta_pol.append(theta_pol[node])
                self.gh_phi.append(phi[node])
                self.gh_r.append(R[node])
                self.gh_z.append(Z[node])
                
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
        if debug: print('Loading ECE Equilibrium Signals')
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        self.Te=conn.get(r'\CMOD::TOP.ELECTRONS:ECE:RESULTS:ECE_TE').data()
        self.R=conn.get(r'dim_of(\CMOD::TOP.ELECTRONS:ECE:RESULTS:ECE_TE,0)').data()
        self.time=conn.get(r'dim_of(\CMOD::TOP.ELECTRONS:ECE:RESULTS:ECE_TE,1)').data()
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
class GPC:
    # ECE radial profile [not high frequency ECE data]
    
    def __init__(self,shotno,debug=True):
        if debug: print('Loading GPC-ECE Signals')
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        baseAddress = r'\CMOD::TOP.ELECTRONS:ECE.GPC_RESULTS:'
        
        self.GPC_freq = conn.get(baseAddress + 'GPC_FREQ').data()
        
        self.Cal = []
        self.Psi = []
        self.Te = []
        self.Rad = []
        for i in np.arange(1,10):
            self.Cal.append(conn.get(baseAddress+'CAL:CAL%d'%i).data())
            #self.Psi.append(conn.get(baseAddress+'PSI_GPC:PSI%d'%i).data())
            self.Te.append(conn.get(baseAddress+'TE:TE%d'%i).data())
            self.Rad.append(conn.get(baseAddress+'RAD:R%d'%i).data())
            
            
        self.Te = np.array(self.Te)
        self.Rad = np.array(self.Rad)
        self.time=conn.get(r'dim_of(\CMOD::TOP.ELECTRONS:ECE:GPC_RESULTS:TE:TE1,0)').data()
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
class GPC_2:
    # ECE radial profile [not high frequency ECE data]
    
    def __init__(self,shotno,debug=True):
        if debug: print('Loading GPC_2-ECE Signals')
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        baseAddress = r'\CMOD::TOP.ELECTRONS:GPC_2:RESULTS:'
        
        self.GPC_freq = conn.get(baseAddress + 'GPC2_FREQ').data()
        
        # self.Cal = []
        # #self.Psi = []
        # self.Te = []
        # self.Rad = []
        
        self.Cal = conn.get(baseAddress+'CALS').data()
        #self.Psi.append(conn.get(baseAddress+'PSI_GPC:PSI%d'%i).data())
        self.Te = conn.get(baseAddress+'GPC2_TE').data()
        self.Rad = conn.get(baseAddress+'RADII').data()
            
            
        #self.Te = np.array(self.Te).squeeze()
        #self.Rad = np.array(self.Rad).squeeze()
        self.time=conn.get(r'dim_of(\CMOD::TOP.ELECTRONS:GPC_2:RESULTS:GPC2_TE0)').data()
        #self.time=conn.get(r'dim_of(\CMOD::TOP.ELECTRONS:ECE:GPC_RESULTS:TE:TE1,0)').data()
        self.Rad_Times =conn.get(r'dim_of(\CMOD::TOP.ELECTRONS:GPC_2:RESULTS:RADII,1)').data()
        conn.closeAllTrees()
    
    def makePlot(self):
        plt.close('ECE-Profile')
        fig,ax=plt.subplots(1,1,num='ECE-Profile',tight_layout=True,figsize=(6,3))
        ax.contourf(self.time,self.Rad[:,50],self.Te,levels=50,zorder=-3,cmap='plasma')
        ax.set_rasterization_zorder(-1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('R [m]')
        norm = Normalize(np.min(self.Te),np.max(self.Te))
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax,
                     label=r'$\mathrm{T_e}$ [keV]')
        plt.show()
                
###############################################################################
class FRCECE:
    # High frequency ECE (FR-C-ECE-F), not availible for shots ~< 2010
    
    def __init__(self,shotno,debug=False):
        if debug: print('Loading FRC-ECE Signals')
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        basePath = r'\CMOD::TOP.ELECTRONS:FRCECE:DATA:'
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
    
    def __init__(self,shotno,debug=False):
        if debug: print('Loading Ip Signal')
        
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        self.ip = -1*conn.get(r'\MAGNETICS::IP').data()
        self.time = conn.get(r'dim_of(\MAGNETICS::IP)').data()
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
    
    def __init__(self,shotno,debug=False):
        if debug: print('Loading ICRF Signal')
        
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
class YAG():
    # Nd:YAG Thomson scattering laser Te, ne profiles
    def __init__(self,shotno,debug=False):
        if debug:  print('Loading Thomson (YAG) Signal')
        
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        # only the old system is availible early on
        old = shotno < 1020000000
        edge = shotno > 1000000000 # edge TS only avilible here and newer
        
        ne_node = r'\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:NE_RZ_T' if old else \
            r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ'
        ne_err_node = r'\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:NE_ERR_ZT' if old else \
            r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_ERR'
        te_node = r'\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:TE_RZ_T' if old else \
            r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_RZ'
        te_err_node = r'\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:TE_ERR_ZT' if old else \
            r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_ERR'
        r_mapped_node = r'\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:R_MID_T' if old else \
            r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T'
        z_node = r'\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:Z_SORTED' if old else \
            r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:Z_SORTED'
        r_node = r'\ELECTRONS::TOP.YAG.RESULTS.PARAM:R' if old else \
            r'\ELECTRONS::TOP.YAG.RESULTS.PARAM:R'  
        
        self.Ne = np.array(conn.get(ne_node).data())
        self.Ne_Err = np.array(conn.get(ne_err_node).data())
        self.Te = np.array(conn.get(te_node).data())
        self.Te_Err = np.array(conn.get(te_err_node).data())
        self.R_Map = np.array(conn.get(r_mapped_node).data())
        self.Z = np.array(conn.get(z_node).data())
        self.R = np.array(conn.get(r_node).data())
        self.time = np.array(conn.get('dim_of('+ne_node+')'))
        
        # If edge exists, pull it
        if edge: 
            self.Ne_Edge = np.array(conn.get(\
                 r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE').data())
            self.Ne_Err_Edge = np.array(conn.get(\
                 r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE:ERROR').data())
            self.Te_Edge = np.array(conn.get(\
                 r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE').data())*1e-3
            self.Te_Err_Edge = np.array(conn.get(\
                 r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:TE:ERROR').data())*1e-3
            self.R_Map_Edge = np.array(conn.get(\
                 r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID').data())
            self.Z_Edge = np.array(conn.get(\
                     r'\ELECTRONS::TOP.YAG_EDGETS.DATA:FIBER_Z').data())
            self.R_Edge = np.array(conn.get(\
                     r'\ELECTRONS::TOP.YAG.RESULTS.PARAM:R').data())
            self.time_Edge = np.array(conn.get(\
                     r'dim_of(\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE)').data())
                
        conn.closeAllTrees()
    
    def return_Profile(self,timePoint,dropChansMain=[3],dropChansEdge=[0,1,2,3]):
        # Select desired R points
        r_inds = np.delete(np.arange(len(self.R_Map)),dropChansMain)
        r_inds_Edge = np.delete(np.arange(len(self.R_Map_Edge)),dropChansEdge)
        
        tInd = np.argmin((self.time-timePoint)**2)
        tInd_Edge = np.argmin((self.time_Edge-timePoint)**2)
        
        # Check if timepoint is valid
        if self.R_Map[0,tInd] < 0: 
            raise SyntaxError('Thomson Data Invalid at %2.2fs'%timePoint)
            
        Te = np.concatenate((self.Te[r_inds,tInd],self.Te_Edge[r_inds_Edge,tInd_Edge]))
        Ne = np.concatenate((self.Ne[r_inds,tInd],self.Ne_Edge[r_inds_Edge,tInd_Edge]))
        R = np.concatenate((self.R_Map[r_inds,tInd],self.R_Map_Edge[r_inds_Edge,tInd_Edge]))
        
        r_sort = np.argsort(R)
        
        return Te[r_sort], Ne[r_sort], R[r_sort]
        
    def makePlot(self,time=1,dropChansMain=[3],dropChansEdge=[0,1,2,3],
                 doSave='',eqdsk=None):
        
        plt.close('Thomson_%d_%1.1f'%(self.shotno,time))
        fig,ax=plt.subplots(1,1,num='Thomson_%d_%1.1f'%(self.shotno,time),
                            tight_layout=True,figsize=(4.5,2.))
        
        tInd = np.argmin((self.time-time)**2)
        r_inds = np.delete(np.arange(len(self.R_Map)),dropChansMain)
        
        r_plot = self.R_Map[r_inds,tInd] if eqdsk is None else \
            eqdsk.rz2psinorm(self.R_Map[r_inds,tInd],0,time,sqrt=True,make_grid=True)[0]
            
        ax.errorbar(r_plot,self.Te[r_inds,tInd],fmt='*',
                    yerr=self.Te_Err[r_inds,tInd])
        ax1=ax.twinx()
        ax1.errorbar(r_plot,self.Ne[r_inds,tInd]*1e-20,\
                     yerr=self.Ne_Err[r_inds,tInd]*1e-20,\
                     c=plt.get_cmap('tab10')(1),fmt='*')
            
        tInd = np.argmin((self.time_Edge-time)**2)
        r_inds_Edge = np.delete(np.arange(len(self.R_Map_Edge)),dropChansEdge)
        
        r_plot_edge = self.R_Map_Edge[r_inds_Edge,tInd] if eqdsk is None else \
            eqdsk.rz2psinorm(self.R_Map_Edge[r_inds_Edge,tInd],0,time,
                             sqrt=True,make_grid=True)[0]
        
        ax.errorbar(r_plot_edge,self.Te_Edge[r_inds_Edge,tInd],fmt='*',\
                    yerr=self.Te_Err_Edge[r_inds_Edge,tInd],c=plt.get_cmap('tab10')(0),\
                        alpha=.7)
        ax1.errorbar(r_plot_edge,self.Ne_Edge[r_inds_Edge,tInd]*1e-20,\
                     yerr=self.Ne_Err_Edge[r_inds_Edge,tInd]*1e-20,\
                     c=plt.get_cmap('tab10')(1),fmt='^',alpha=.7)
            
        p1=ax.plot(.8,1,'k*',label='TS Core',ms=3)
        p2=ax.plot(.8,1,'k^',label='TS Edge',ms=1)
        ax.legend(fontsize=8,title='%d: %1.1f s'%(self.shotno,time),
                  title_fontsize=9,loc='lower left')
        p1[0].remove();p2[0].remove()
        yl1 = ax.get_ylim();ax.set_ylim([0,yl1[1]])
        yl2 = ax1.get_ylim();ax1.set_ylim([0,yl2[1]])
        ax.grid()
        ax.set_xlabel('R [m]' if eqdsk is None else r'$\sqrt{\psi_n}$')
        ax.set_ylabel(r'T$_\mathrm{e}$ [keV]')
        ax1.set_ylabel(r'n$_\mathrm{e}$ [$10^{20}\,\mathrm{m}^{-3}$]')
        
        if doSave:fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',
                              transparent=True)
        plt.show()
        
        return ax,ax1
###############################################################################
class POWER_SYSTEM():
    # Equilibrium coil currents, automatically handles smoothing (some signals noisy)
    def __init__(self,shotno,debug=False,doSmoothing=1e3):
        if debug:  print('Loading Equilibrium Coil Current Signals')
        

        
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        base = r'\ENGINEERING::TOP.POWER_SYSTEM.'
        
        
        self.time = conn.get(r'dim_of('+ base + 'TF.CUR_4)').data()
        
        
        self.EF1_L = doFilter(conn.get(base + 'EF1_L.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        self.EF1_U = doFilter(conn.get(base + 'EF1_U.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        
        self.EF2_L = doFilter(conn.get(base + 'EF2_L.A_CUR').data(),\
                                self.time,False,doSmoothing )[0]
        self.EF2_U = doFilter(conn.get(base + 'EF2_U.A_CUR').data(),\
                                self.time,False,doSmoothing )[0]
        
        self.EF3 = doFilter(conn.get(base + 'EF3.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        #self.EF3_U = conn.get(base + 'EF1_L.CUR').data()
        
        self.EF4 = doFilter(conn.get(base + 'EF4.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        #self.EF4_U = conn.get(base + 'EF1_L.CUR').data()
        
        
        self.EFC_A = doFilter(conn.get(base + 'EFC.A_CUR').data(),\
                                self.time,False,doSmoothing )[0] * 1e-3
        self.EFC_B = doFilter(conn.get(base + 'EFC.B_CUR').data(),\
                                self.time,False,doSmoothing )[0] * 1e-3
        self.EFC_C = doFilter(conn.get(base + 'EFC.C_CUR').data(),\
                                self.time,False,doSmoothing )[0] * 1e-3
        
        
        self.OH1 = doFilter(conn.get(base + 'OH1.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        self.OH2_U = doFilter(conn.get(base + 'OH2_L.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        self.OH2_L = doFilter(conn.get(base + 'OH2_U.CUR').data(),\
                                self.time,False,doSmoothing )[0]
        
        
        self.TF_1 = doFilter(conn.get(base + 'TF.CUR_1').data(),\
                                self.time,False,doSmoothing )[0]
        self.TF_2 = doFilter(conn.get(base + 'TF.CUR_2').data(),\
                                self.time,False,doSmoothing )[0]
        self.TF_3 = doFilter(conn.get(base + 'TF.CUR_3').data(),\
                                self.time,False,doSmoothing )[0]
        self.TF_4 = doFilter(conn.get(base + 'TF.CUR_4').data(),\
                                self.time,False,doSmoothing )[0]
        
    
        conn.closeAllTrees()
        
        

    def makePlots(self,doSave=False):
        plt.close('Coil_Currents_%d'%self.shotno)
        fig,ax=plt.subplots(2,2,num='Coil_Currents_%d'%self.shotno,
                tight_layout=True,sharex=True,sharey=False,figsize=(4,4))
        
        ax[0,1].sharey(ax[0,0])
        ax[1,1].sharey(ax[1,0])
        
        ax[0,0].plot(self.time,self.EF1_L,label='EF1_L',alpha=.7)
        ax[0,0].plot(self.time,self.EF1_L,label='EF1_U',alpha=.7)
        ax[0,0].plot(self.time,self.EF1_L,label='EF2_L',alpha=.7)
        ax[0,0].plot(self.time,self.EF1_L,label='EF2_U',alpha=.7)
        ax[0,0].plot(self.time,self.EF1_L,label='EF3',alpha=.7)
        ax[0,0].plot(self.time,self.EF1_L,label='EF4',alpha=.7)
        
        ax[0,1].plot(self.time,self.EFC_A,label='EFC_A',alpha=.7)
        ax[0,1].plot(self.time,self.EFC_B,label='EFC_B',alpha=.7)
        ax[0,1].plot(self.time,self.EFC_C,label='EFC_C',alpha=.7)
        
        ax[1,0].plot(self.time,self.OH1,label='OH1',alpha=.7)
        ax[1,0].plot(self.time,self.OH2_U,label='OH2_U',alpha=.7)
        ax[1,0].plot(self.time,self.OH2_L,label='OH2_L',alpha=.7)
        
        ax[1,1].plot(self.time,self.TF_1,label='TF_1',alpha=.7)
        ax[1,1].plot(self.time,self.TF_2,label='TF_2',alpha=.7)
        ax[1,1].plot(self.time,self.TF_3,label='TF_3',alpha=.7)
        ax[1,1].plot(self.time,self.TF_4,label='TF_4',alpha=.7)
        
        for i in range(2):
            for j in range(2):
                if i==0 and j==1:ax[i,j].legend(loc='upper right',fontsize=8,
                            handlelength=1,title=self.shotno,title_fontsize=9)
                else: ax[i,j].legend(loc='upper right',fontsize=8,handlelength=1)
                ax[i,j].grid()
                if j==0:ax[i,j].set_ylabel('Current [kA]')
                if i==1:ax[i,j].set_xlabel('Time [s]')
                if j==1:plt.setp(ax[i,j].get_yticklabels(), visible=False)
    
        plt.show()
        
        if doSave:fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',
                              transparent=True)
        
        
###############################################################################
class A_EQDSK_CCBRSP():
    # Coil currents in kAmp-Turns, from aEqsdsk file, for simulation
    # Order is specific: OH1, OH2U, OH2L EF1U, EF1L, EF2U, EF2L, EF3U, EF3L,
    # EF4U, EF4L, EFCU, EFCL, TFTU, TFTL
    def __init__(self,shotno,debug=False):
        
        if debug:  print('Loading Equilibrium Coil Current Signals from aEQDSK File')
        
        conn = openTree(shotno)
        self.shotno=shotno if shotno !=0 else currentShot(conn)
        
        conn = openTree(shotno)
        self.currs_all = conn.get(r'\cmod::top.mhd.analysis:efit.results.a_eqdsk.ccbrsp').data()*1e-3
        self.currs_time = conn.get(r'dim_of(\cmod::top.mhd.analysis:efit.results.a_eqdsk.ccbrsp,1)').data()
    
    def saveOutput(self,tPoint,saveFile='../data_output/'):
        tInd = np.argmin((self.currs_time-tPoint)**2)
        
        if saveFile: 
            names = [ 'OH1', 'OH2U', 'OH2L', 'EF1U', 'EF1L', 'EF2U', 'EF2L',
                 'EF3U', 'EF3L', 'EF4U', 'EF4L', 'EFCU', 'EFCL', 'TFTU', 'TFTL']
            np.savetxt(saveFile+'aEqdsk_Currs_%d_%1.2f.txt'%(self.shotno,tPoint),\
               np.array([names, self.currs_all[tInd]],dtype=object).T,fmt='%s, %4.4e' )
        return self.currs_all[tInd]
    
    
    def makePlots(self,doSave=False):
        plt.close('Coil_Currents_aEQDSK_%d'%self.shotno)
        fig,ax=plt.subplots(2,2,num='Coil_Currents_aEQDSK_%d'%self.shotno,
                tight_layout=True,sharex=True,sharey=False,figsize=(4,4))
        
        #ax[0,1].sharey(ax[0,0])
        #ax[1,1].sharey(ax[1,0])
        
        ax[0,0].plot(self.currs_time,self.currs_all[:,3],label='EF1U',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,4],label='EF1L',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,5],label='EF2U',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,6],label='EF2L',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,7],label='EF3U',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,8],label='EF3L',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,9],label='EF4U',alpha=.7)
        ax[0,0].plot(self.currs_time,self.currs_all[:,10],label='EF4L',alpha=.7)
        
        ax[0,1].plot(self.currs_time,self.currs_all[:,11],label='EFCU',alpha=.7)
        ax[0,1].plot(self.currs_time,self.currs_all[:,12],label='EFCL',alpha=.7)
        
        ax[1,0].plot(self.currs_time,self.currs_all[:,0],label='OH1',alpha=.7)
        ax[1,0].plot(self.currs_time,self.currs_all[:,1],label='OH2U',alpha=.7)
        ax[1,0].plot(self.currs_time,self.currs_all[:,2],label='OH2U',alpha=.7)
        
        ax[1,1].plot(self.currs_time,self.currs_all[:,13],label='TFTU',alpha=.7)
        ax[1,1].plot(self.currs_time,self.currs_all[:,14],label='TFTL',alpha=.7)
        
        for i in range(2):
            for j in range(2):
                if i==0 and j==1:ax[i,j].legend(loc='upper right',fontsize=8,
                            handlelength=1,title=self.shotno,title_fontsize=9)
                else: ax[i,j].legend(loc='upper right',fontsize=8,handlelength=1)
                ax[i,j].grid()
                if j==0:ax[i,j].set_ylabel('Current [kA-turns]')
                if i==1:ax[i,j].set_xlabel('Time [s]')
                #if j==1:plt.setp(ax[i,j].get_yticklabels(), visible=False)
    
        #plt.show()
        
        if doSave:fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',
                              transparent=True)
        plt.show()
        
###############################################################################
class TRANSP():
    # Pull Transp Data required for FAR 3D
    # Note: TRANSP tree shot numberings different than regular tree
    
    def __init__(self, shotno, debug=False):
         if debug: print('Loading electon/ion density/temperature, fast ion density, pressure')
         
         self.shotno = shotno
         
         conn = openTree(shotno,'transp')
         
         self.output={}
         self.output['ni_m3_rho_tor'] = conn.get(r'\NI').data() * 1e6
         self.output['ne_m3_rho_tor'] = conn.get(r'\NE').data() * 1e6
         
         self.output['Ti_eV_rho_tor'] = conn.get(r'\TI').data()
         self.output['Te_eV_rho_tor'] = conn.get(r'\TE').data()
         
         self.output['nMinority_m3_rho_tor'] = conn.get(r'\NMINI').data()* 1e6 
         # Note: there may not be an explicit fast ion density accessable
         UFPAR_M = conn.get(r'\UMINPA').data() * 1e6 # MINORITY PLL ENERGY DENSITY
         UFPRP_M = conn.get(r'\UMINPP').data() * 1e6 # MINORITY PERP ENERGY DENSITY

         U_FI_PAR = conn.get(r'\UFASTPA').data() * 1e6 # Fast ion parallel energy density
         U_FI_PP = conn.get(r'\UFASTPP').data() * 1e6
         
         self.output['PMinority_Pa_rho_tor'] = 2/3 * (UFPAR_M + UFPRP_M)
         self.output['PFastIon_Pa_rho_tor'] = 2/3 * (U_FI_PAR + U_FI_PP)
         
         # x"r/a" ctr, X(TIME3, X) square root of normalized toroidal flux in the "center of the zone"
         self.X = conn.get('\X').data()[0,:]
         self.time = conn.get(r'dim_of(\X,1)').data()
         
    def save_output(self,timePoint,outFile='../data_output/',debug=False):
        # Write output profiles to JSON file 
        
        tInd = np.argmin((self.time-timePoint)**2)
        outDat = self.output.copy()
        
        outDat['ni_m3_rho_tor'] = self.output['ni_m3_rho_tor'][tInd].tolist()
        outDat['ne_m3_rho_tor'] = self.output['ne_m3_rho_tor'][tInd].tolist()
        
        outDat['Ti_eV_rho_tor'] = self.output['Ti_eV_rho_tor'][tInd].tolist()
        outDat['Te_eV_rho_tor'] = self.output['Te_eV_rho_tor'][tInd].tolist()
        
        outDat['nMinority_m3_rho_tor'] = self.output['nMinority_m3_rho_tor'][tInd].tolist()
        
        outDat['PMinority_Pa_rho_tor'] = self.output['PMinority_Pa_rho_tor'][tInd].tolist()
        outDat['PFastIon_Pa_rho_tor'] = self.output['PFastIon_Pa_rho_tor'][tInd].tolist()
        
        outDat['X'] = self.X.tolist()
        outDat['time'] = self.time[tInd].tolist()
        
        with open(outFile+'TRANSP_Tree_%d_%1.1f.json'%(self.shotno,timePoint),\
                  'w', encoding='utf-8') as f: 
            json.dump(outDat,f, ensure_ascii=False, indent=4)
        
        if debug: print('Saved: '+outFile+'TRANSP_Tree_%d_%1.1f.json'%(self.shotno,timePoint))
        
        return outDat
    def makePlots2D(self,doSave=''):
        
        plt.close('Transp_Output_%d'%self.shotno)
        fig,ax = plt.subplots(2,4,sharex=True,sharey=True,tight_layout=True,\
                          num='Transp_Output_%d'%self.shotno,figsize=(9,3.5))
        
        vmin=np.min(self.output['ne_m3_rho_tor']);vmax=np.max(self.output['ne_m3_rho_tor'])
        norm = Normalize(vmin*1e-20,vmax*1e-20)
        ax[0,0].contourf(self.time,self.X,self.output['ne_m3_rho_tor'].T,
                         vmin=vmin,vmax=vmax,levels=20,cmap='plasma',zorder=-3)
        ax[1,0].contourf(self.time,self.X,self.output['ni_m3_rho_tor'].T,
                         vmin=vmin,vmax=vmax,levels=20,cmap='plasma',zorder=-3)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax[0,0],
                     location='top',label=r'[$10^{20}\,\mathrm{m}^{-3}$]')
        ax[0,0].text(.05,.76,r"n$_\mathrm{e}$",transform=ax[0,0].transAxes,
                     bbox=dict(facecolor='white',alpha=.3,color='white'))
        ax[1,0].text(.05,.8,r"n$_\mathrm{i}$",transform=ax[1,0].transAxes,
                     bbox=dict(facecolor='white',alpha=.3,color='white'))
        
        vmin=np.min(self.output['Te_eV_rho_tor']);vmax=np.max(self.output['Te_eV_rho_tor'])
        norm = Normalize(vmin*1e-3,vmax*1e-3)
        ax[0,1].contourf(self.time,self.X,self.output['Te_eV_rho_tor'].T,
                         vmin=vmin,vmax=vmax,levels=20,cmap='plasma',zorder=-3)
        ax[1,1].contourf(self.time,self.X,self.output['Ti_eV_rho_tor'].T,
                         vmin=vmin,vmax=vmax,levels=20,cmap='plasma',zorder=-3)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax[0,1],
                     location='top',label=r'[keV]')
        ax[0,1].text(.05,.76,r"T$_\mathrm{e}$",transform=ax[0,1].transAxes,
                     bbox=dict(facecolor='white',alpha=.3,color='white'))
        ax[1,1].text(.05,.8,r"T$_\mathrm{i}$",transform=ax[1,1].transAxes,
                     bbox=dict(facecolor='white',alpha=.3,color='white') )
        
        c=ax[1,2].contourf(self.time,self.X,self.output['nMinority_m3_rho_tor'].T*1e-19,
                         levels=20,cmap='plasma',zorder=-3)
        fig.colorbar(c,ax=ax[0,2],
                     location='top',label=r'n$_\mathrm{fi}$ [$10^{19}\,\mathrm{m}^{-3}$]')
        
        c=ax[1,3].contourf(self.time,self.X,self.output['PMinority_Pa_rho_tor'].T*1e-3,
                         levels=20,cmap='plasma',zorder=-3)
        fig.colorbar(c,ax=ax[0,3],
                     location='top',label=r'P$_\mathrm{fi}$ [kJ m$^{-3}$]')
        ax[1,3].text(.05,.8,'%d'%self.shotno,transform=ax[1,3].transAxes,
                     bbox=dict(facecolor='white',alpha=.3,color='white') )
        
        for i in range(2):ax[0,2+i].set_axis_off()
        
        for i in range(4):
            for j in range(2):
                if j==1:ax[j,i].set_xlabel('Time [s]')
                if i==0:ax[j,i].set_ylabel(r'$\sqrt{\psi_n}$')
                ax[j,i].set_rasterization_zorder(-1)
        
        if doSave: fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',
                               transparent=True)
        
        plt.show()
    
    def makePlots1D(self,timePoint,ax=None,ax1=None,doSave=''):
        
        tInd = np.argmin((self.time-timePoint)**2)
        
        flagRedoAxes = False
        if ax is None:
            plt.close('Transp_Output_%d_%1.1f'%(self.shotno,timePoint))
            fig,ax=plt.subplots(1,1,tight_layout=True,figsize=(4,2),
                                num='Transp_Output_%d_%1.1f'%(self.shotno,timePoint))
            flagRedoAxes = True
        ax.plot(self.X,self.output['Te_eV_rho_tor'][tInd,:]*1e-3,'-',
                c=plt.get_cmap('tab10')(0),alpha=.6)
        if ax1 is None: ax1 = ax.twinx()
        ax1.plot(self.X,self.output['ne_m3_rho_tor'][tInd,:]*1e-20,'-',\
                 alpha=.6,c=plt.get_cmap('tab10')(1))
        
        if flagRedoAxes:
            ax.set_xlabel(r'$\sqrt{\psi_n}$')
            ax.set_ylabel(r'T$_\mathrm{e}$ [keV]')
            ax1.set_ylabel(r'n$_\mathrm{e}$ [$10^{20}\mathrm{m}^{-3}$]')
            ax.grid()
        else:
            l=ax1.plot([.9,.9],[.1,.1],'-k',label='EFIT',alpha=.6,lw=1)
            ax1.legend(loc='upper right',fontsize=8,handlelength=1)
            l[0].remove()
        
        if doSave:
            plt.gcf().savefig(doSave+plt.gcf().canvas.manager.get_window_title()+'_efit.pdf',
                        transparent=True)
        plt.show()
        
###############################################################################

class _get_wtor():
    # Gets solid-body toroidal rotation freqeuncy
    def __init__(self, shot=None, tht=0,):

        # Obtains data tree
        tree = MDSplus.Tree('spectroscopy', shot)
    
        # Initializes output
        self.dwtor = {}
    
        # ------------------
        # Obtains HIREXSR z line data
        # ------------------
        self.dwtor['HIREXSR_z'] = {}
    
        nlab = r'\spectroscopy::top.hirexsr.analysis'
        if tht > 0:
            nlab += '%i'%(tht)
        print('test')
        try:
            # Obtain ion temp data
            self.dwtor['HIREXSR_z']['val_kHz'] = tree.getNode(
                nlab + '.helike.profiles.z:pro'
                #r'\spectroscopy::top.hirexsr.analysis.helike.profiles.z:pro'
                ).data()[1,:,:].T # [kHz], dim(nchan,t)
            print('test')
            # Obtain ion temp errorbar data
            self.dwtor['HIREXSR_z']['err_kHz'] = tree.getNode(
                nlab + '.helike.profiles.z:proerr'
                #r'\spectroscopy::top.hirexsr.analysis.helike.profiles.z:proerr'
                ).data()[1,:,:].T # [kHz], dim(nchan,t)
    
            # Radial domain
            self.dwtor['HIREXSR_z']['psin'] = tree.getNode(
                nlab + '.helike.profiles.z:rho'
                #r'\spectroscopy::top.hirexsr.analysis.helike.profiles.z:rho'
                ).data().T # [], dim(mchan,t)
    
            # Time domain
            self.dwtor['HIREXSR_z']['t_s'] = tree.getNode(
                nlab + '.helike.profiles.z:rho'
                #r'\spectroscopy::top.hirexsr.analysis.helike.profiles.z:rho'
                ).dim_of(0).data() # [s], dim(t,)
            
            # Error check
            if self.dwtor['HIREXSR_z']['val_kHz'].shape[0] > self.dwtor['HIREXSR_z']['psin'].shape[0]:
                ss = self.dwtor['HIREXSR_z']['psin'].shape[0]
                self.dwtor['HIREXSR_z']['val_kHz'] = self.dwtor['HIREXSR_z']['val_kHz'][:ss,:]
                self.dwtor['HIREXSR_z']['err_kHz'] = self.dwtor['HIREXSR_z']['err_kHz'][:ss,:]
    
        # In case this diag is not available
        except:
            print('HIREXSR z line for wtor not available')
    
        # ------------------
        # Obtains HIREXSR Lya1 line data
        # ------------------
        self.dwtor['HIREXSR_Lya1'] = {}
    
        try:
            # Obtain ion temp data
            self.dwtor['HIREXSR_Lya1']['val_kHz'] = tree.getNode(
                nlab + '.hlike.profiles.lya1:pro'
                #r'\spectroscopy::top.hirexsr.analysis.hlike.profiles.lya1:pro'
                ).data()[1,:,:].T # [kHz], dim(nchan,t)
    
            # Obtain ion temp errorbar data
            self.dwtor['HIREXSR_Lya1']['err_kHz'] = tree.getNode(
                nlab + '.hlike.profiles.lya1:proerr'
                #r'\spectroscopy::top.hirexsr.analysis.hlike.profiles.lya1:proerr'
                ).data()[1,:,:].T # [kHz], dim(nchan,t)
    
            # Radial domain
            self.dwtor['HIREXSR_Lya1']['psin'] = tree.getNode(
                nlab + '.hlike.profiles.lya1:rho'
                #r'\spectroscopy::top.hirexsr.analysis.hlike.profiles.lya1:rho'
                ).data().T # [], dim(mchan,t)
    
            # Time domain
            self.dwtor['HIREXSR_Lya1']['t_s'] = tree.getNode(
                nlab + '.hlike.profiles.lya1:rho'
                #r'\spectroscopy::top.hirexsr.analysis.hlike.profiles.lya1:rho'
                ).dim_of(0).data() # [s], dim(t,)
    
        # In case this diag is not available
        except:
            print('HIREXSR Lya1 line for wtor not available')
    
        # Output
        #return dwtor

###############################################################################
###############################################################################
# Local data storage functionality
def __loadData(shotno,data_archive='',debug=True,forceReload=False,\
               pullData = ['bp','bp_t','gpc','ip','p_rf','yag']):
    # data_archive can be manually specified if the default file locaiton isn't in use
    # Defult is to the author's MFE directory
    if data_archive == '': data_archive = data_archive_path
    
    
    if debug: print('Attempting to load: ' + \
                        data_archive + 'rawData_%d.pk'%shotno)
        
    rawData = {} # Initialize empty data container
    try: 
        rawData = pk.load(open(data_archive + 'rawData_%d.pk'%shotno,'rb'))
        
        # If we need to reload something, or need a signal not already saved
        if forceReload or not np.all(np.isin(pullData,list(rawData.keys()))): raise Exception
    except:
        rawData = __genRawData(rawData,shotno,pullData,debug)
        __saveRawData(rawData,shotno,debug,data_archive)
        
    if debug:print('Loaded ' + 'rawData_%d.pk'%shotno +' from archives')
    
    return rawData
###################################################
def __genRawData(rawData,shotno,pullData,debug):
    # Pull requested diagnostic signals
    
    
    if 'bp' in pullData and 'bp' not in rawData: rawData['bp'] = BP(shotno,debug)
    
    if 'bp_t' in pullData and 'bp_t' not in rawData: rawData['bp_t'] = BP_T(shotno, debug)
    
    if 'ece' in pullData and 'ece' not in rawData: rawData['ece'] = ECE(shotno,debug)
    
    if 'gpc' in pullData and 'gpc' not in rawData: rawData['gpc'] = GPC(shotno,debug)
    
    if 'gpc_2' in pullData and 'gpc_2' not in rawData: rawData['gpc_2'] = GPC_2(shotno,debug)
    
    if 'frcece' in pullData and 'frcece' not in rawData: rawData['frcece'] = FRCECE(shotno, debug)
    
    if 'ip' in pullData and 'ip' not in rawData: rawData['ip'] = Ip(shotno,debug)
    
    if 'p_rf' in pullData and 'p_rf' not in rawData: rawData['p_rf'] = RF_PWR(shotno,debug)
    
    if 'yag' in pullData and 'yag' not in rawData: rawData['yag'] = YAG(shotno,debug)
    
    return rawData

###################################################
def __saveRawData(rawData,shotno,debug=False,data_archive=''):
    if data_archive == '': data_archive = data_archive_path
    
    with open(data_archive+'rawData_%d.pk'%shotno,'wb') as f:pk.dump(rawData,f)
    
    if debug: print('Successfully saved: ' + data_archive+'rawData_%d.pk'%shotno)
