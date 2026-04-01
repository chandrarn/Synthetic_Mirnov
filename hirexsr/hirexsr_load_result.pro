;in IDL, first @hirexsr_ini.bat
@/mnt/home/cgao/idl/graphics/greek.pro
@/mnt/home/cgao/idl/interp2d/interp2d.pro
pro hirexsr_load_result,shot,line=line,tht=tht,profile,moment,lineint,ti_fit=ti_fit,omega_fit=omega_fit
;+
; NAME: hirexsr_load_result
;
; AUTHOR:
;   Chi Gao, PSFC/MIT
;   cgao@mit.edu
;
; PURPOSE:
;   This routine is written to load the result of hirexsr.
;
; DESCRIPTION: three results: profiles, moment, line intergrated profile              
;    
; INPUT:
;       shot: cmod shot number
;        tht: tht node for hirex. Default is 0
;       line: line for hirex.1---w line. 2---z line. 3---H-like Lya1 line
;
; OUTPUT:
;      profile: {prof,proferr[,prof1,proferr1],tau,psi_norm,r/a,Rmid,Rmagx,a}    (if m=1 term exits, prof1 and proferr1 will carry it)
;               profile.prof:{emiss[a.u.],Vtor[km/s],Vpol[km/s],Ti[keV],Omega[rad/s],Omega/2!pi[kHz]}   
;               profile.proferr:{emiss_err,Vtor_err,Vpol_err,Ti_err,Omega_err,Omega/2!pi_err}
;               profile.prof1:{emiss[a.u.],Vtor[km/s],Vpol[km/s],Ti[keV],Omega[rad/s],Omega/2!pi[kHz]}
;               profile.proferr1:{emiss_err,Vtor_err,Vpol_err,Ti_err,Omega_err,Omega/2!pi_err}     
;      moment: {mom,momerr,tau}
;              moment.mom:{0th, 1st, 2nd}
;              moment.momerr:{0th, 1st, 2nd}
;      lineint: {lint,linterr,tau,psi_norm,r/a,Rmid}
;            lineint.lint:{emiss[a.u.],Vtor[km/s],Vpol[km/s],Ti[keV],Omega[rad/s],Omega/2!pi[kHz]}
;            lineint.linterr:{emiss_err,Vtor_err,Vpol_err,Ti_err,Omega_err,Omega/2!pi_err} 
;
;
;      ti_fit: if keyword set to 1, ti profiles will be spline fitted
;      and stored in profiles
;
;
;
;      omega_fit: if keyword set to 1, omega profiles will be spline
;      fitted and stored in profiles
;
; HISTORY:
;   05/22/2013: Created by cgao. The profiles are functions of psi. To
;   get r/a information, I need efit mapping..
;   09/23/2014: efit_rmid conflicts with /usr/local/cmod/codes/efit/idl/efit_rmid.pro . So I renamaed it to efit_rmidd


;===================================================================================================
;======================================== Load EFIT ================================================
;===================================================================================================
mdsopen,'analysis',shot
efit_tau=mdsvalue('dim_of(\efit_aeqdsk:rmagx)')        ; time 
efit_rmidd=transpose(mdsvalue('\ANALYSIS::EFIT_RMID')) ; Rmid
efit_psi= mdsvalue('dim_of(\ANALYSIS::EFIT_RMID,1)') ; psi values of different rmid values
efit_rmagx=mdsvalue('\efit_aeqdsk:rmagx')/100.  ;R of magnetic axis
efit_aout=mdsvalue('\efit_aeqdsk:aout')/100.    ;r of lcfs, 0 at geometric center.
efit_rout=mdsvalue('\efit_aeqdsk:rout')/100     ;R of geometric center
efit_a=efit_rout+efit_aout-efit_rmagx ;Actual plasma minor radius
mdsclose
efit_roa=efit_rmidd*0
for it=0,n_elements(efit_tau)-1 do efit_roa(*,it)=(efit_rmidd(*,it)-efit_rmagx(it))/efit_a(it)
help,efit_psi,efit_roa,efit_tau


;===================================================================================================
;======================================== Load ti_fit ================================================
if keyword_set(ti_fit) then begin
   print,'example ti_fit file name: /home/cgao/fits/tifit_1110217032_THT0.dat'
   filename=''
   read,filename,prompt='Enter file name(with path):'
   while(!file_test(filename)) do begin
      print,'file not exist. Try again'
      read,filename,prompt='Enter file name(with path):'
   endwhile 
   restore,filename=filename,/v
   ;help,bsti,time
   tmp=*bsti(0)
   nt_fit=n_elements(time)
   nr_fit=n_elements(tmp.fit.rho)
   nr_fit_raw=n_elements(tmp.dat.rho)
 ; the structure of bsom is difficult to read...
 ; tmp.fit: prof,rho,err,dprof,derr,good. What is the rho here?
 ;          the rho here should be the same as labeled in tmp.dat.rlab
 ; tmp.dat:om,err,rho,type,rlab. Here rho should be r/a, or the same as rlab
   
   ti_fit_profile=fltarr(nr_fit,nt_fit)
   ti_fit_profile_err=fltarr(nr_fit,nt_fit)
   ti_fit_rho=tmp.fit.rho
   ti_raw_profile=fltarr(nr_fit_raw,nt_fit)
   ti_raw_profile_err=fltarr(nr_fit_raw,nt_fit)
   ti_raw_rho=tmp.dat.rho
   ti_rho_label=a.dat.rlab
   for it=0,nt-1 do begin
      tmp=*bsti(it)
      ti_fit_profile(*,it)=tmp.fit.prof
      ti_fit_profile_err(*,it)=tmp.fit.err
      ti_raw_profile(*,it)=tmp.dat.ti
      ti_raw_profile_err(*,it)=tmp.dat.err
   endfor
endif


;===================================================================================================
;======================================== Load omega_fit ================================================

if keyword_set(omega_fit) then begin
   print,'example omega_fit file name: /home/cgao/fits/omfit_1110217032_THT0.dat'
   filename=''
   read,filename,prompt='Enter file name(with path):'
   while(!file_test(filename)) do begin
      print,'file not exist. Try again'
      read,filename,prompt='Enter file name(with path):'
   endwhile 
   restore,filename=filename,/v
   ;help,bsom,time
   tmp=*bsom(0)
   nt_fit=n_elements(time)
   nr_fit=n_elements(tmp.fit.rho)
   nr_fit_raw=n_elements(tmp.dat.rho)
 ; the structure of bsom is difficult to read...
 ; tmp.fit: prof,rho,err,dprof,derr,good. What is the rho here?
 ;          the rho here should be the same as labeled in tmp.dat.rlab
 ; tmp.dat:om,err,rho,type,rlab. Here rho should be r/a, or the same as rlab
   
   om_fit_profile=fltarr(nr_fit,nt_fit)
   om_fit_profile_err=fltarr(nr_fit,nt_fit)
   om_fit_rho=tmp.fit.rho
   om_raw_profile=fltarr(nr_fit_raw,nt_fit)
   om_raw_profile_err=fltarr(nr_fit_raw,nt_fit)
   om_raw_rho=tmp.dat.rho
   om_rho_label=a.dat.rlab
   for it=0,nt-1 do begin
      tmp=*bsom(it)
      om_fit_profile(*,it)=tmp.fit.prof
      om_fit_profile_err(*,it)=tmp.fit.err
      om_raw_profile(*,it)=tmp.dat.om
      om_raw_profile_err(*,it)=tmp.dat.err
   endfor
endif

;===================================================================================================
;======================================== Load Profiles ================================================
;      profile: {prof,proferr[,prof1,proferr1],tau,psi_norm,r/a,Rmid,Rmagx,a}    (if m=1 term exits, prof1 and proferr1 will carry it)
;               profile.prof:{emiss[a.u.],Vtor[km/s],Vpol[km/s],Ti[keV],Omega[rad/s],Omega/2!pi[kHz]}   
;               profile.proferr:{emiss_err,Vtor_err,Vpol_err,Ti_err,Omega_err,Omega/2!pi_err}
;               profile.prof1:{emiss[a.u.],Vtor[km/s],Vpol[km/s],Ti[keV],Omega[rad/s],Omega/2!pi[kHz]}
;               profile.proferr1:{emiss_err,Vtor_err,Vpol_err,Ti_err,Omega_err,Omega/2!pi_err}    
;      As of 03/20/2014, the poloidal profile is not implemented so it is always 0

;===================================================================================================
hirexsr_load_profile,shot,line,pro_arr,proerr_arr,rho_arr,tau,status=status,tgood=tgood,tht=tht,tinst=tinst
psi=rho_arr(*,0)
tau=tau(where(tgood eq 1))



if psi((n_elements(psi)-1)/2) lt psi((n_elements(psi)-1)/2+1) then begin ; no m=1 term
   roa=interp2d(efit_roa,efit_psi,efit_tau,psi,tau,grid=1) ;map efit_roa to hirex time and psi radius
   rmid=interp2d(efit_rmidd,efit_psi,efit_tau,psi,tau,grid=1)  ;map efit_rmidd to hirex time and psi radius
   rmagx=interpol(efit_rmagx,efit_tau,tau) ;map efit_rmagx to hirex time
   a=interpol(efit_a,efit_tau,tau)  ;map efit_a to hirex time   
   prof = {   $
             emiss:pro_arr(*,where(tgood eq 1),0), $
             vtor:pro_arr(*,where(tgood eq 1),1)*2*!pi*rmid, $
             vpol:pro_arr(*,where(tgood eq 1),2), $
             ti:pro_arr(*,where(tgood eq 1),3)-tinst,$
             Omega:pro_arr(*,where(tgood eq 1),1)*2*!pi , $
             freq:pro_arr(*,where(tgood eq 1),1) $
             }

   proferr = {   $
             emiss_err:proerr_arr(*,where(tgood eq 1),0), $
             vtor_err:proerr_arr(*,where(tgood eq 1),1)*2*!pi*rmid, $
             vpol_err:proerr_arr(*,where(tgood eq 1),2), $
             ti_err:proerr_arr(*,where(tgood eq 1),3),$
             Omega_err:proerr_arr(*,where(tgood eq 1),1)*2*!pi , $
             freq_err:proerr_arr(*,where(tgood eq 1),1) $
             }
   profile={prof:prof,proferr:proferr,tau:tau,psinorm:psi,rnorm:roa,Rmid:rmid,Rmagx:rmagx,a:a}
endif else begin ; there is m=1 term
   psi=psi(0:(n_elements(psi)-1)/2)
   roa=interp2d(efit_roa,efit_psi,efit_tau,psi,tau,grid=1)   ;map efit_roa to hirex time and psi radius
   rmid=interp2d(efit_rmidd,efit_psi,efit_tau,psi,tau,grid=1)  ;map efit_rmidd to hirex time and psi radius
   rmagx=interpol(efit_rmagx,efit_tau,tau) ;map efit_rmagx to hirex time
   a=interpol(efit_a,efit_tau,tau)  ;map efit_a to hirex time   
   prof = {   $
             emiss:pro_arr(0:(n_elements(psi)-1),where(tgood eq 1),0), $
             vtor:pro_arr(0:n_elements(psi)-1,where(tgood eq 1),1)*2*!pi*rmid, $
             vpol:pro_arr(0:n_elements(psi)-1,where(tgood eq 1),2), $
             ti:pro_arr(0:n_elements(psi)-1,where(tgood eq 1),3)-tinst,$
             Omega:pro_arr(0:n_elements(psi)-1,where(tgood eq 1),1)*2*!pi , $
             freq:pro_arr(0:n_elements(psi)-1,where(tgood eq 1),1) $
             }

   proferr = {   $
             emiss_err:proerr_arr(0:n_elements(psi)-1,where(tgood eq 1),0), $
             vtor_err:proerr_arr(0:n_elements(psi)-1,where(tgood eq 1),1)*2*!pi*rmid, $
             vpol_err:proerr_arr(0:n_elements(psi)-1,where(tgood eq 1),2), $
             ti_err:proerr_arr(0:n_elements(psi)-1,where(tgood eq 1),3),$
             Omega_err:proerr_arr(0:n_elements(psi)-1,where(tgood eq 1),1)*2*!pi , $
             freq_err:proerr_arr(0:n_elements(psi)-1,where(tgood eq 1),1) $
             }
   prof1 = {   $
             emiss:pro_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),0), $
             vtor:pro_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),1)*2*!pi*rmid, $
             vpol:pro_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),2), $
             ti:pro_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),3)-tinst,$
             Omega:pro_arr(n_elements(psi)+1:2*n_elements(psi)-1,where(tgood eq 1),1)*2*!pi , $
             freq:pro_arr(n_elements(psi)+1:2*n_elements(psi)-1,where(tgood eq 1),1) $
             }

   proferr1 = {   $
             emiss_err:proerr_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),0), $
             vtor_err:proerr_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),1)*2*!pi*rmid, $
             vpol_err:proerr_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),2), $
             ti_err:proerr_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),3),$
             Omega_err:proerr_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),1)*2*!pi , $
             freq_err:proerr_arr(n_elements(psi):2*n_elements(psi)-1,where(tgood eq 1),1) $
             }
   profile={prof:prof,proferr:proferr,prof1:prof1,proferr1:proferr1,tau:tau,psinorm:psi,rnorm:roa,Rmid:rmid,Rmagx:rmagx,a:a}
endelse


;===================================================================================================
;======================================== Load Moments ================================================
;      moment: {mom,momerr,tau}
;              moment.mom:{0th, 1st, 2nd}
;              moment.momerr:{0th, 1st, 2nd}
;===================================================================================================
hirexsr_load_momentptr,shot,line,momptr,tau,pos,tpos,lam_o,z,dlam=dlam,status=status,tht=tht,tree=tree
good=where(tau ne -1)
tau=tau(good)
momptr=momptr(good)
dum=*momptr(0)
nchannel=(size(dum))(1)
mom=fltarr(nchannel,3,n_elements(tau))
momerr=fltarr(nchannel,3,n_elements(tau))
for it=0,n_elements(tau)-1 do begin	     &$  
   dum=*momptr(it)	     &$  
   mom(*,*,it)=dum(*,0:2)	     &$  
   momerr(*,*,it)=dum(*,3:5)	     &$  
   endfor
moment={mom:mom,momerr:momerr,tau:tau}

;===================================================================================================
;======================================== Load LineInt ================================================
;      lineint: {lint,linterr,tau,psi_norm,r/a,Rmid}
;            lineint.lint:{emiss[a.u.],Vtor[km/s],Vpol[km/s],Ti[keV],Omega[rad/s],Omega/2!pi[kHz]}
;            lineint.linterr:{emiss_err,Vtor_err,Vpol_err,Ti_err,Omega_err,Omega/2!pi_err} 
;===================================================================================================
hirexsr_load_mlintptr,shot,line,mlint,tau,tht=tht
good=where(tau ne -1)
tau=tau(good)
mlint=mlint(good)
dum=*mlint(0)
nchannel=(size(dum))(1)
emiss=fltarr(nchannel,n_elements(tau))
vtor=fltarr(nchannel,n_elements(tau))
vpol=fltarr(nchannel,n_elements(tau))
ti=fltarr(nchannel,n_elements(tau))
omega=fltarr(nchannel,n_elements(tau))
freq=fltarr(nchannel,n_elements(tau))
emiss_err=fltarr(nchannel,n_elements(tau))
vtor_err=fltarr(nchannel,n_elements(tau))
vpol_err=fltarr(nchannel,n_elements(tau))
ti_err=fltarr(nchannel,n_elements(tau))
omega_err=fltarr(nchannel,n_elements(tau))
freq_err=fltarr(nchannel,n_elements(tau))
psi=fltarr(nchannel,n_elements(tau))
roa=fltarr(nchannel,n_elements(tau))
rmid=fltarr(nchannel,n_elements(tau))
rmagx=fltarr(n_elements(tau))
a=fltarr(n_elements(tau))
for it=0,n_elements(tau)-1 do begin	     &$  
   dum=*mlint(it)	     &$  
   psi(*,it)=dum(*,4)    &$  
   roa(*,it)=dum(*,10)    &$  
   rmid(*,it)=dum(*,5)    &$
   a(it)=(rmid(nchannel-1,it)-rmid(0,it))/(roa(nchannel-1,it)-roa(0,it))    &$
   rmagx(it)=rmid(0,it)-a(it)*roa(0,it)    &$
   emiss(*,it)=dum(*,8)	     &$  
   emiss_err(*,it)=dum(*,9)	     &$  
   vtor(*,it)=dum(*,0)*2*!pi*rmid(*,it)            &$  
   vtor_err(*,it)=dum(*,1)*2*!pi*rmid(*,it)	     &$  
   vpol(*,it)=dum(*,6)	     &$  
   vpol_err(*,it)=dum(*,7)	     &$
   omega(*,it)=dum(*,0)*2*!pi      &$  
   omega_err(*,it)=dum(*,1)*2*!pi  &$     
   freq(*,it)=dum(*,0)             &$  
   freq_err(*,it)=dum(*,1)         &$     
   ti(*,it)=dum(*,2)	             &$  
   ti_err(*,it)=dum(*,3)	     &$  
   endfor
lint={emiss:emiss,vtor:vtor,vpol:vpol,ti:ti,Omega:omega,freq:freq}
linterr={emiss_err:emiss_err,vtor_err:vtor_err,vpol_err:vpol_err,ti_err:ti_err,Omega_err:omega_err,freq_err:freq_err}
lineint={lint:lint,linterr:linterr,tau:tau,psinorm:psi,rnorm:roa,Rmid:rmid,Rmagx:rmagx,a:a}
end
