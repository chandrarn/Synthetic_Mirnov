from MDSplus import *
import re
import numpy

class Mirnov :
    '''
    This class is a wrapper for Mirnov coil data for a discharge,
    including signals, as well as location information.
    
    It will slowly incorporate useful signal processing tools.
    
    It is written for Python3, but should be compatible for Python2....
    
    T. Golfinopoulos, 13 March 2018
    '''
    
    def __init__(self,s,t1=None,t2=None,coil_selection=None):
        '''
        Initialize object
        
        USAGE:
            m=Mirnov(s[,t1][,t2][,coil_selection])
        
        INPUT:
            s=shot number
            t1=Optional input.  First time in timebase, default=earliest available time.
            t2=Optional input.  Last time in timebase, default=earliest available time.
            coil_selection=Optional input.  String input:
               'all' => all available coils are pulled
               'tile' => coils in arrays under main limiter tiles
               are pulled - names like BP1T_GHK
               'low-n' => coils on extensions (in low-n coil
               array) are pulled - names like BP_BC_TOP
               'pol' => coils in poloidal array are pulled
               a regular expression => coil names are filtered
               with this regular expression to pull a subset
               list => list of strings corresponding
               to coil names that should be pulled.
               Default='all'
            
        OUTPUT:
            m=Mirnov object caching data for indicated coils between given time range of given shot.
        
        T. Golfinopoulos, 13 March 2018
        '''
        self.s=s
        
        self.tree=Tree('magnetics',s)
        #return 
        tt=None
        
        if(t1 is None) :
            try :
                self.t1=self.tree.getNode('active_mhd.dt216_start').getData().evaluate()
            except :
                tt=self.tree.getNode('active_mhd.signals.bp1t_ghk').getData().getDimensionAt(0).evaluate().getData().evaluate() #Really need to get that recursive evaluate...jeez...
                self.t1=tt[0]
        else :
            self.t1=t1
        
        if(t2 is None) :
            if not(tt is None) :
                self.t2=tt[-1]
            else :
                tt=self.tree.getNode('active_mhd.signals.bp1t_ghk').getData().getDimensionAt(0).evaluate().getData().evaluate() #Really need to get that recursive evaluate...jeez...
                self.t2=tt[-1]
        else :
            self.t2=t2
        
        if coil_selection is None :
            self.coil_selection='all'
        elif type(coil_selection)==list :
            self.coil_selection=[n.lower() for n in coil_selection]
        else :
            self.coil_selection=lower(coil_selection)
        
        #NO: test=self.Tree.getNodeWild('active_mhd.signals.*') #Returns empty list
        #NO: self.Tree.getNodeWild('active_mhd.signals.bp***') #Returns all sub nodes of magnetics signal nodes.
        magNodes=self.tree.getNode('active_mhd.signals').getNodeWild('bp*','signal')
        #Only keep nodes that are on.
        coilNames=[n.getNodeName().lower() for n in magNodes]
        keepNodes=[]

        for n in magNodes :
            if not n.isOn() :
                continue
            elif 'OLD' in n.getNodeName():continue# discount these nodes
            elif self.coil_selection=='tile' and re.match('.*t.*',n.getNodeName().lower()) is None:
                continue
            elif self.coil_selection=='low-n' and re.match('.*top|.*bot',n.getNodeName().lower()) is None:
                continue
            elif self.coil_selection=='pol' and re.match('bp\d\d_..k',n.getNodeName().lower()) is None:
                continue
            elif type(coil_selection)==str :
                if re.match(coil_selection,n.getNodeName().lower()) is None :
                    continue
            elif type(coil_selection)==list :
                if not n.getNodeName().lower() in coil_selection :
                    continue
            else :
                keepNodes.append(n)          
        
        self.nodes=keepNodes
        self.coil_names=[n.getNodeName().lower() for n in self.nodes]
        
        self.name2node=dict(zip(self.coil_names,self.nodes)) #Dictionary pulling node from name
        self.name2name=dict(zip(self.coil_names,self.coil_names)) #Convenience 1:1 mapping in dictionary so same calling sequence, getAttr, can be used to retrieve coil names from regular expression or list.
        
        #Pull info for coils - R, z, phi, theta
        #Start with linked list.  Then, also generate dictionaries.

        #Cast to list so that concatenation of lists will work.
        phi = list(self.tree.getNode('rf_lim_coils.phi_ab').getData().evaluate().getData().evaluate())+list(self.tree.getNode('rf_lim_coils.phi_gh').getData().evaluate().getData().evaluate())
        nodeNames=list(self.tree.getNode('rf_lim_coils.nodename').getData().evaluate()[0:(len(phi))]);
        try: # Catch for: low-n sensors move around 2008
            nodeNames += list(self.tree.getNode('low_n_coils.nodename').getData().evaluate().getData().evaluate())
        except:pass
        #Deblank (trim whitespace) the strings and enforce lower case
        nodeNames=[n.strip().lower() for n in nodeNames]
        
        
        
        z = list(self.tree.getNode('rf_lim_coils:z_ab').getData().evaluate().getData().evaluate())+list(self.tree.getNode('rf_lim_coils:z_gh').getData().evaluate().getData().evaluate())
        
        R = list(self.tree.getNode('rf_lim_coils:r_ab').getData().evaluate().getData().evaluate())+list(self.tree.getNode('rf_lim_coils:r_gh').getData().evaluate().getData().evaluate())
        
        theta=list(self.tree.getNode('rf_lim_coils:theta_pol_ab').getData().evaluate().getData().evaluate())+list(self.tree.getNode('rf_lim_coils:theta_pol_gh').getData().evaluate().getData().evaluate())
        
        try: 
            z += list(self.tree.getNode('low_n_coils.z').getData().evaluate().getData().evaluate())
            R += list(self.tree.getNode('low_n_coils.r').getData().evaluate().getData().evaluate())
            phi += list(self.tree.getNode('low_n_coils.phi').getData().evaluate().getData().evaluate())
            theta += list(self.tree.getNode('low_n_coils.theta_pol').getData().evaluate().getData().evaluate())
        except: 
            # _k coils were replaced by the _top, _bot coils around 2008
            z_k= self.tree.getNode('RF_LIM_COILS.Z_K').getData().evaluate().getData().evaluate()
            r_k=self.tree.getNode('RF_LIM_COILS.R_K').getData().evaluate().getData().evaluate()
            phi_k=self.tree.getNode('RF_LIM_COILS.PHI_K').getData().evaluate().getData().evaluate()
            theta_k=[0,0,0,0,0,0]#self.tree.getNode('RF_LIM_COILS.THETA_POL_K').getData().evaluate().getData().evaluate()
            
            for ind,name in enumerate([['bp0%d_k'%i] for i in numpy.arange(1,7)]):
                z += [z_k[ind]]
                R += [r_k[ind]]
                phi += [phi_k[ind]]
                theta += [theta_k[ind]]  
                nodeNames += name
        
        name2phi=dict(zip(nodeNames,phi))
        name2z=dict(zip(nodeNames,z))
        name2R=dict(zip(nodeNames,R))
        name2theta=dict(zip(nodeNames,theta))
        
        
        #Store R,z,phi,theta for subset of coils specified,
        #and in order of self.nodes and self.coil_names
        self.phi=[name2phi[n] for n in self.coil_names]
        self.z=[name2z[n] for n in self.coil_names]
        self.R=[name2R[n] for n in self.coil_names]
        self.theta=[name2theta[n] for n in self.coil_names]
        
        #Create dictionaries mapping name to info
        self.name2phi=dict(zip(self.coil_names,self.phi))
        self.name2z=dict(zip(self.coil_names,self.z))
        self.name2R=dict(zip(self.coil_names,self.R))
        self.name2theta=dict(zip(self.coil_names,self.theta))
        
        #Initialize empty lists for signals
        self.num_sigs_pulled=0 #Initialize counter for number of signals pulled
        self.num_raw_sigs_pulled=0 #Initialize counter for number of signals pulled
        self._sigsPulled=False #Flip to true when all signals pulled
        self._rawSigsPulled=False #Flip to true when all signals pulled
        self.name2sig=dict.fromkeys(self.coil_names) #Generate dictionary of None values
        self.name2raw_sig=dict.fromkeys(self.coil_names) #Generate dictionary of None values
        
        #Pull timebases -- 
        min_t=self.t1
        max_t=self.t2
        self.keep_inds=None #If interpolating onto time base, don't need array of indices to keep (masking)
        try :
            #Try to pull nodes - they may not be available for some shots.
            nt=[self.tree.getNode('active_mhd.signals.timebase1'),
                    self.tree.getNode('active_mhd.signals.timebase2'),
                    self.tree.getNode('active_mhd.signals.timebase3')]
            
            #Evaluate timebases and synchronize.
            all_t=[]
            ii=0
            self.dt=None
            self.n_timebases=len(nt) #3 timebases - need to interpolate later
            for n in nt :
                all_t.append(numpy.array(n.getData().evaluate().getData()))
                
                #Get intersection of time range with t1 and t2
                if min_t<min(all_t[ii]) :
                    min_t=min(all_t[ii])

                if max_t>max(all_t[ii]) :
                    max_t=max(all_t[ii])
                
                this_dt=all_t[ii][1]-all_t[ii][0]
                if self.dt is None :
                    self.dt=this_dt
                    self.t=all_t[ii][numpy.array(all_t[ii]>=min_t) & numpy.array(all_t[ii]<=max_t)]
                elif self.dt < this_dt :
                    self.dt=this_dt
                    self.t=all_t[ii][numpy.array(all_t[ii]>=min_t) & numpy.array(all_t[ii]<=max_t)]
                    
                ii+=1
            
            #Calculate sampling frequency
            self.fs=1.0/self.dt
            
        except :
            self.n_timebases=1 #Only 1 timebase - don't need to interpolate later
            print('timebase1, timebase2, and timebase3 nodes not available--proceeding with timebase...')
            try :
                self.t=numpy.array(self.tree.getNode('active_mhd.signals.timebase').getData().evaluate().getData())
            except :
                print('active_mhd.signals.timebase node also unavailable -- pull for bp1t_ghk')
                self.t=numpy.array(self.tree.getNode('active_mhd.signals.bp1t_ghk').getDimensionAt(0).getData().evaluate().getData())
            
            min_t=max(self.t[0],min_t)
            max_t=min(self.t[-1],max_t)
            self.keep_inds=numpy.array(self.t>=min_t) & numpy.array(self.t<=max_t)
            self.t=self.t[self.keep_inds]
        self.t1=min_t
        self.t2=max_t

            
    def getTree(self) :
        '''
        Return MDSplus tree for this Mirnov instance.
        
        USAGE:
            myTree=mirnov_instance.getTree()
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.tree
        
    def getAttr(self,this_attr,coil_list=None):
        '''
        Helper function to pull attributes for optional list of coils
        
        INPUT:
            this_attr=dictionary of name/value pairs, name=coil names, value can be toroidal angle (phi), R, z, theta, etc. (coil attributes).
             coil_list=optional input to determine coil subset for which data should be pulled.  Default=all 
                        coils.
        
        USAGE:
            phis=mirnov_instance.getAttr(self.name2phi) #Return toroidal angles for all coils
            phis=mirnov_instance.getAttr(self.name2phi,reg_exp) #Return phi for coils whose name matches regular expression
            phis=mirnov_instance.getPhi(self.name2phi,coil_list) #List of names for which to return angles, in same order
            
        OUTPUT:
            List of given attribute values corresponding to coil list.  If there is only a single match, a single value is returned, else a list is returned.
            
        T. Golfinopoulos, 24 Mar. 2018
        '''
        if coil_list is None:
            #Return attributes of all coils 
            return list(this_attr.values)
        elif type(coil_list) is list :
            out=[] #Initialize empty list
            for coil in coil_list :
                out.append(this_attr[coil]) #Append values in same order as given in list
        else :
            if type(coil_list) is str : #Make sure coil_list is string
                out=[] #Initialize empty list
                for coil_name in this_attr.keys() :
                    #Add values whose corresponding coil name (key) matches the given regular expression
                    if (re.match(coil_list,coil_name)) :
                        out.append(this_attr[coil_name])
            else :
                print("Input, coil_list, must be a string")
                return None

        if len(out)==1 :
            #If only one element, don't return list
            return out[0]
        else :
            return out #Return list of attributes

    def getPhi(self,coil_list=None) :
        '''
        Return toroidal angles in degrees (relative to Port A, increasing with counterclockwise rotation when viewed from top) for subset of Mirnov coils.
        
        INPUT:
            coil_list=optional input to determine coil subset for which data should be pulled.  Default=all 
                        coils.
        
        USAGE:
            phis=mirnov_instance.getPhi() #Return toroidal angles for all coils
            phis=mirnov_instance.getPhi(reg_exp) #Return phi for coils whose name matches regular expression
            phis=mirnov_instance.getPhi(coil_list) #List of names for which to return angles, in same order
        
        If there is only a single match, a single value is returned; else a list is returned
        
        See also getAttr, getZ, getR, getNodes, getTheta
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.getAttr(self.name2phi,coil_list)
            
    def getZ(self,coil_list=None) :
        '''
        Return vertical heights in meters for a subset of Mirnov coils.
        See getPhi, getR, getTheta, getNodes, getAttr
        
        USAGE:
            zs=mir_inst.getZ() #Return all z's
            zs=mir_inst.getZ(reg_exp) #Regular expression input to filter coil names
            zs=mir_inst.getZ(coil_list) #List of names - z's returned in same order as list
        
        If there is only a single match, a single value is returned; else a list is returned
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.getAttr(self.name2z,coil_list)

    def getR(self,coil_list=None) :
        '''
        Return major radii in meters for a subset of Mirnov coils.
        See getPhi, getZ, getTheta, getNodes, getAttr
        
        USAGE:
            Rs=mir_inst.getR() #Return all R's
            Rs=mir_inst.getR(reg_exp) #Regular expression input to filter coil names
            Rs=mir_inst.getR(coil_list) #List of names - R's returned in same order as list
        
        If there is only a single match, a single value is returned; else a list is returned
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.getAttr(self.name2R,coil_list)

    def getTheta(self,coil_list=None) :
        '''
        Return poloidal orientation in degrees for a subset of Mirnov coils.
        See getPhi, getZ, getR, getNodes, getAttr
        
        USAGE:
            thetas=mir_inst.getTheta() #Return all theta's
            thetas=mir_inst.getTheta(reg_exp) #Regular expression input to filter coil names
            thetas=mir_inst.getTheta(coil_list) #List of names - thetas returned in same order as list
        
        If there is only a single match, a single value is returned; else a list is returned
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.getAttr(self.name2theta,coil_list)

    def getNodes(self, coil_list) :
        '''
        Return the MDSplus tree nodes corresponding to a given subset of Mirnov coils.
        See also getPhi, getZ, getR, getTheta, getAttr
        
        USAGE:
            nodes=mir_inst.getNodes() #Return all nodes
            nodes=mir_inst.getNodes(reg_exp) #Regular expression input to filter coil names
            nodes=mir_inst.getNodes(coil_list) #List of names - nodes returned in same order as list
        
        If there is only a single match, a single value is returned; else a list is returned
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.getAttr(self.name2node, coil_list)
    
    def getT1(self):
        '''
        Return t1, start time in which signals are truncated.
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.t1

    def getT2(self):
        '''
        Return t2, end time in which signals are truncated.
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return self.t2

    def getTRange(self):
        '''
        Return list, [t1,t2], start and end times in which signals are truncated.
        
        T. Golfinopoulos, 24 Mar. 2018
        '''
        return [self.t1,self.t2]
    
    def getSig(self, coil_list, force_reload=False, raw_sig=False):
        '''
        Return signal values for corresponding nodes for given subset of Mirnov coils.  Truncate in given time range, t1 to t2
        
        USAGE:
            signals=mir_inst.getSig() #Return signals between t1 and t2
            signals=mir_inst.getSig(reg_exp) #Return signals between t1 and t2 for coils with names filtered by regular expression, reg_exp
            signals=mir_inst.getSig(coil_list) #Return signals between t1 and t2 for coils whose name are in list, coil_list, with signals in same order as coil names in list
            signals=mir_inst.getSig(...,force_reload=True) #Force signals to be reloaded
            signals=mir_inst.getSig(...,raw_sig=True) #Pull raw data, without frequency-dependent calibration
        
        If there is only one match to reg_exp or coilList, return signal values for just this match (i.e. output is one-dimensional numpy array)
        Otherwise, return a list of signals (output is list of numpy arrays)
        
        See also getNodes, getAttr
        T. Golfinopoulos, 24 Mar. 2018
        '''
        names=self.getAttr(self.name2name,coil_list)
        if not type(names) is list :
            names=[names] #Make sure names is a list of strings
        
        #Only pull data for requested signals so that don't need to pull all signals
        #every time.  But cache data so that don't have to pull every time.
        #When all signals have been pulled, don't need to check any more.
        if raw_sig :
            sig_check=self._rawSigsPulled
        else :
            sig_check=self._sigsPulled
            
        if not sig_check or force_reload:
            #Fetch signals if this has not been done yet
            for key in names :
                this_node=self.name2node[key]
                val=self.name2sig[key]
                if raw_sig :
                    try :
                        #Get raw subnode
                        this_node=this_node.getNode('raw')
                        val=self.name2raw_sig[key]
                    except :
                        print("No raw node exists for this shot, "+str(self.s))
                        return None

                if val is None :
                    #If data has not been pulled for this signal, then pull data
                    this_y=numpy.array(this_node.getData().evaluate().getData())
                    this_t=numpy.array(this_node.getDimensionAt(0).getData().evaluate().getData())
                    
                    #Align on uniform timebase between t1 and t2
                    if self.n_timebases>1 :
                        #Interpolate onto timebase
                        this_y=numpy.interp(self.t,this_t,this_y)
                    else :
                        this_y=this_y[self.keep_inds]
                    
                    if raw_sig :
                        self.name2raw_sig[key]=this_y
                        self.num_raw_sigs_pulled+=1 #Increment count on number of signals pulled
                    else :
                        self.name2sig[key]=this_y
                        self.num_sigs_pulled+=1 #Increment count on number of signals pulled
            
            if(not raw_sig and self.num_sigs_pulled==len(self.coil_names)) :
                #Flip bit indicating signals have been pulled
                self._sigsPulled=True
            elif(raw_sig and self.num_raw_sigs_pulled==len(self.coil_names)) :
                #Flip bit indicating signals have been pulled
                self._rawSigsPulled=True
        
        if raw_sig :
            return self.getAttr(self.name2raw_sig,coil_list)
        else :
            return self.getAttr(self.name2sig,coil_list)

    def getT(self) :
        '''
        Return the timebase - this is constructed to be in the time range 
        specified for this Mirnov coil object at instantiation, with the 
        smallest available sampling time available from the timebases (up to  
        three for the C-Mod Mirnov coil system, given the fact that the D-tAcq
        digitizers cannot reliably produce identical databases for identical
        control settings, per the information of our synchronization system). 
        All signals returned by getSig() are on this timebase, either by a
        logical mask (i.e. match indices given a single timebase across all 
        coils) or by interpolation (when there are is more than one timebase
        across the Mirnov coils).
        
        T. Golfinopoulos, 24 March 2018
        '''
        
        return self.t