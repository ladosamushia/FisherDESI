
# coding: utf-8

# In[1]:

from numpy import array, log, exp, zeros, size, linspace, loadtxt, pi, sqrt, copy, dot, transpose, diag, ones
from numpy.linalg import pinv, det, inv
from scipy.integrate import quad
from scipy.interpolate import interp1d
from itertools import product, combinations_with_replacement

"""
This program will compute projected errors on parameters fs8 (growth rate), 
alpha_parallel (line-of-sight BAO ~ H), alpha_perpendicular (across-the-line-of-sight BAO ~ D_A),
expected from a sample of multiple populations of galaxies over a specified redshift range. 
This is a Fisher information based prediction assuming that the power spectrum measurement will be analyzed 
up to a certain wavenumber -- k_max.
The shape of the power spectrum in physical units is assumed to be known perfectly from CMB measurements.
This implements a standard algorithm as in e.g. Simpson & Peacock (arXiv:0910.3834) and many others.

The program requires three input files and generates one output file.

First input file (ifile) should have the columns -- zmin, zmax, bias1, nbar1, bias2, nbar2, ...
and as many rows as there are redshift bins.
zmin, zmax -- upper and lower boundaries of a redshift bin.
bias -- bias of the sample. This is a bias without any factors of G(z) (b^2 = P_gg/P_mm)
nbar -- number density of tracers in units of dN/dz/d(sq.deg.)
There should be one pair of bias, nbar values for each sample in that redshift bin.
All columns should have the same length so if one of your samples does not cover the entire redshift range
just put zero bias and zero nbar in that row.

Second input file (Psmooth) should have two columns -- k, Pk
k -- wavenumber in units of h/Mpc
Pk -- power spectrum in unites of (Mpc/h)^3
This file describes the shape of the "smooth" power spectrum and is normalized
in such a way that the Plinear from which it was generated has s8=1

The third input file (Pbao) also has two columns -- k, Pk
k -- wavenumber in units of h/Mpc (doesnt really have to be the same sampling as Psmooth).
Pk -- BAO only power spectrum (dimensionless).
For the definition of Pbao and Psmooth see e.g. Anderson et al. (arXiv:1312.4877)

The format of the output file is self-explanatory.

When practical, all efforts have been made to use exactly the same assumptions 
as in Font-Ribera et al. (arXiv:1308.4164).
Major differences from Font-Ribera et al.:
a) The shape of the primordial power spectrum is fixed. Variations in the shape due to uncertainties
in cosmological parameters are assumed to be negligible.
b) The code only computes constraints from DESI (or any other galaxy survey). Does not combine with CMB
to get a combined FOM.

The output of this codes matches the results in DESIdocI (arXiv:1611.00036) when the same fiducial cosmology 
is used. The default cosmological parameters and Pk provided for this code are however in Planck best-fit cosmology
so the results slightly differ from the ones in DESIdocI.

Lado Samushia (colkhis@gmail.com), February 2017.
"""

# ====================================================
# Here are the parameters that you may want to change 
# ====================================================
# Choose one of the options depending on whether or not you want
# The "BAO only" or the "Full Shape" results.
BAOonly = True
#BAOonly = False
# Input and output file names
ifile = "desi_samples.txt"
psmoothfile = "Psmooth.txt"
Pbaofile = "Pbao.txt"
ofile = "desi_predictions.txt"
# Speed of light in 1e5 times m/s
clight = 2997.92458 
# Footprint in sq.deg.
Area = 14000
# Maximum wavenumber considered in the analysis
kmax = 0.2
# Reconstruction factor. How well will the reconstruction work depending on nP(k=0.16 h/Mpc, mu=0.6)
# E.g. 0.0 means a perfect reconstruction, 0.5 means 50% of the nonlinear degradation will be reconstructed, 
# 1.0 means reconstructino does not work.
# The assumptions, methodology, and the numerical values for r_factor are from Font-Ribera et al.
nP = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0]
r_factor = [1.0, 0.9, 0.8, 0.70, 0.6, 0.55, 0.52, 0.5]
# Fiducial cosmological parameters
Om = 0.3175
Ok = 0
Ol = 1 - Om - Ok
s8 = 0.8
w0 = -1
wa = 0
h = 0.67
parfid = [w0, wa, Om, Ok, h]
# =======================================================

# ============================
# Functions
# ============================


def volume(zmin,zmax):
    """Volume of a redshift shell in (Mpc/h)^3"""
    dist = lambda z: 1/sqrt(Om*(1+z)**3 + Ol)*clight
    rmax = quad(dist, 0, zmax)[0]
    rmin = quad(dist, 0, zmin)[0]
    return Area*(rmax**3 - rmin**3)*4*pi/3

def H(z,par):
    """Hubble parameter in w0waCDMGR in km/Mpc/s"""
    w0,wa,Om,Ok,h = par
    return 100*h*sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + (1-Om-Ok)*(1+z)**(3*(1+w0))*exp(3*wa*z/(1+z)))

def DA(z,par):
    """The angular distance in Mpc"""
    da = quad(lambda x: 1/H(x,par),0,z)[0]
    da *= clight/(1+z)
    return da

def G(z,par):
    """The growth factor (LCDMGR). Normalized so that G(z=0) = 1"""
    Gz = H(z,par)*quad(lambda x: (1+x)/H(x,par)**3,z,10000)[0]
    G0 = H(0,par)*quad(lambda x: (1+x)/H(x,par)**3,0,10000)[0]
    return Gz/G0

def f(z,par):
    """The growth rate = dlnG/dlna (LCDMGR)"""
    dz = 0.01
    lnG2 = log(G(z+dz,par))
    lnG1 = log(G(z,par))
    return -(lnG2-lnG1)/dz*(1+z)

def Sigma_per(z,par):
    """The parameter describing nonlinear degradation of the perpendicular BAO in (Mpc/h)"""
    return 9.4*s8/0.9*G(z,par)

def Sigma_par(z,par):
    """The parameter describing nonlinear degradation of the parallel BAO in (Mpc/h)"""
    return (1 + f(z,par))*Sigma_per(z,par)

def Pk2d(k,mu,z,p1,p2,par):
    """2D galaxy power spectrum in (Mpc/h)^3. Accounting for nonlinear degradation in the BAO"""
    # CAUTION this is the power spectrum with the shot noise term 1/n
    zz = red[z]
    if nbar[z,p1]==0 or nbar[z,p2]==0:
        return 0
    f, apar, aper = par[-3:]
    b1 = par[p1]
    b2 = par[p2]
    kr = k*sqrt((1-mu**2)/aper**2 + mu**2/apar**2)
    mur = k*mu/apar/kr
    # I use mu and k here (instead of mur and kr) because I don't want the damping terms
    # To contribute to the fisher matrix (Font-Ribera et al. make the same assumption).
    Dpar = mu**2*k**2*Spar[z]**2
    Dper = (1-mu**2)*k**2*Sper[z]**2
    #
    # 
    Dfactor = exp(-rfact[z]**2*(Dpar + Dper)/2)
    if BAOonly == True:
        Pnl = ((Pbao(kr) - 1)*Dfactor + 1)*Psmooth(k)
        power = (b1 + mu**2*f)*(b2 + mu**2*f)*Pnl*s8**2
    else:
        # Font-Ribera say we should multiply the overall shape by Dfactor 
        # as it rougly describes the loss of information in from the full shape due to nonlinearities as well
        Pnl = ((Pbao(kr) - 1)*Dfactor + 1)*Psmooth(kr)*Dfactor
        power = (b1 + mur**2*f)*(b2 + mur**2*f)*Pnl*s8**2   
    if p1 == p2:
        power += 1/nbar[z,p1]
    return power

def Pk2dlin(k,mu,z,p1,p2,par):
    """2D power spectrum of galaxies but without nonlinear degradation in the BAO"""
    # CAUTION this is the power spectrum with the shot noise term 1/n
    zz = red[z]
    if nbar[z,p1]==0 or nbar[z,p2]==0:
        return 0
    f, apar, aper = par[-3:]
    b1 = par[p1]
    b2 = par[p2]
    kr = k*sqrt((1-mu**2)/aper**2 + mu**2/apar**2)
    mur = k*mu/apar/kr
    power = (b1 + mur**2*f)*(b2 + mur**2*f)*Pbao(k)*Psmooth(k)*s8**2
    if p1 == p2:
        power += 1/nbar[z,p1]
    return power

def CovP(k,mu,z,p1,p2,p3,p4,par):
    """Covariance of two power spectra <P_12 P_34>, where indeces 1234 refer to (in principle different) galaxy samples"""
    # I don't account for the nonlinear BAO in the covariance. This is faster and does not really make a big difference
    # Font-Ribera et al. make the same assumption.
    covar = (Pk2dlin(k,mu,z,p1,p3,par)*Pk2dlin(k,mu,z,p2,p4,par) + Pk2dlin(k,mu,z,p1,p4,par)*Pk2dlin(k,mu,z,p2,p3,par))/2
    return covar

def dPdb(k,mu,z,p1,p2,par):
    """Derivative (numerical) of P_12 with respect to parameters"""
    P1 = Pk2d(k,mu,z,p1,p2,par)
    dPdp = zeros(Npar)
    for i in range(Npar):
        pr2 = copy(par)
        dp = pr2[i]/100
        pr2[i] += dp
        P2 = Pk2d(k,mu,z,p1,p2,pr2)
        pr2[i] -= dp
        if dp == 0:
            dPdp[i] = 0
        else:
            dPdp[i] = (P2 - P1)/dp
    return dPdp

# Footprint as fraction of the full sky
Area /= 41253

# k-binning
Nk = 25
kmax = 0.2
kbin = linspace(0, kmax, Nk+1)
kbin = (kbin[1:] + kbin[:-1]) / 2
dk = kbin[1] - kbin[0]
# mu-binning
Nmu = 25
mubin = linspace(0, 1, Nmu+1)
mubin = (mubin[1:] + mubin[:-1]) / 2
dmu = mubin[1] - mubin[0]
# I have checked that 25 bins in k and mu are sufficiently accurate

# Load input file and determine redshift bins, biases, and number densities
JJ = loadtxt(ifile)
zmin = JJ[:,0]
zmax = JJ[:,1]
red = (zmin + zmax)/2
nbar = JJ[:,3::2]
bias = JJ[:,2::2]
# Number of galaxy populations
Npop = (size(JJ,axis=1) - 2)//2
# Number of possible cross and auto power spectra
Npk = Npop*(Npop+1)//2
# Number of redshift bins
Nz = size(JJ,axis=0)
        
# Number of parameters 
# The ordering is [b_1, b_2, ..., f, apar, aper]
Npar = 3 + Npop
Par = zeros(Npar)
# Covariance matrix of power spectra
CPP = zeros((Npk,Npk))
# Fisher matrix of power spectra
FPP = zeros((Npk,Npk))
#Derivatives of Pk with respect to parameters
derP = zeros((Npar,Npk))

# Fisher matrix of parameters [b_1, b_2, ..., f, apar, aper].
Fbb = zeros((Npar*Nz,Npar*Nz))
#Derivatives of parameters with respect to [w0,wa,Om,Ok,h]
dbdw = zeros((5,Npar*Nz))

# Fisher matrix of [w0, wa, Om, Ok, h]
Fww = zeros((5,5))

# Smooth linear matter power spectrum normilized to s8=1
km, pm = loadtxt("Psmooth.txt",unpack=True)
Psmooth = interp1d(km,pm,kind='cubic')
# Linear BAO only power spectrum
km, pm = loadtxt("Pbao.txt",unpack=True)
Pbao = interp1d(km,pm,kind='cubic')

# Precompute the nonlinear degradation parameters in each redshift slice for speed
Spar = zeros(Nz)
Sper = zeros(Nz)
for z in range(Nz):
    Spar[z] = Sigma_par(red[z],parfid)
    Sper[z] = Sigma_per(red[z],parfid)
    
# Renormalize nbar from per deg^2*dz to per (Mpc/h)^3
for i in range(Nz):
    nbar[i,:] *= Area*4*pi*(180/pi)**2/volume(zmin[i],zmax[i])*(zmax[i]-zmin[i])

#
# Depending on the number density of tracers see how well the reconstruction will work at different redshits
#
RF = interp1d(nP,r_factor)
# The combined nP of all the tracers
nPD = zeros(Nz)
rfact = zeros(Nz)
for z in range(Nz):
    zz = red[z]
    for i in range(Npop):
        parforP = [bias[z,i],bias[z,i],f(zz,parfid)*G(zz,parfid),1,1]
        if nbar[z,i] != 0:
            nPD[z] += nbar[z,i]*Pk2dlin(0.16,0.6,z,i,i,parforP)
    if nPD[z] < nP[0]:
        rfact[z] = r_factor[0]
    elif nPD[z] > nP[-1]:
        rfact[z] = r_factor[-1]
    else:
        rfact[z] = RF(nPD[z])
        
# Fisher of power spectra
for z in range(Nz):
    Vol = volume(zmin[z],zmax[z])
    zz = red[z]
    print("z = {0:.2f}, V = {1:.2e} (Gpc/h)^3".format(zz,Vol/1e9))
    Par[:Npop] = bias[z,:]*G(zz,parfid)
    Par[-3:] = [f(zz,parfid)*G(zz,parfid), 1, 1]
    print("fs8(z) fiducial: {0:.2f}".format(f(zz,parfid)*G(zz,parfid)*s8))
    # Loop over k and mu bins
    for index in product(kbin,mubin):
        k, m = index
        ps1 = 0
        # Loop over power spectra of different samples P_12
        for pair1 in combinations_with_replacement(range(Npop),2):
            n1, n2 = pair1
            ps2 = 0
            # Loop over power spectra of different samples P_34
            for pair2 in combinations_with_replacement(range(Npop),2):
                n3, n4 = pair2
                # Cov(P_12,P_34)
                CPP[ps1,ps2] = CovP(k,m,z,n1,n2,n3,n4,Par)
                ps2 += 1
            derP[:,ps1] = dPdb(k,m,z,n1,n2,Par)
            ps1 += 1
        FPP = pinv(CPP)*Vol/(2*pi)**2*k**2*dk*dmu
        Fbb[z*Npar:(z+1)*Npar,z*Npar:(z+1)*Npar] += dot(derP,dot(FPP,transpose(derP)))

Cbb = pinv(Fbb)
# Diagonal errors.
# I put abs here because sometimes numerical zeros fluctuate to very small negative numbers.
# This is because the Fbb has a zero determinant and we use pinv.
sigma_bb = sqrt(diag(abs(Cbb)))

out_file = open(ofile,"w")
print("z","sigma_fs8 (%)", "sigma_H (%)", "sigma_DA (%)")
out_file.write("# z sigma_fs8(%) sigma_H (%) sigma_DA(%)\n")
for z in range(Nz):
    zz = red[z]
    fs8fid = f(zz,parfid)*G(zz,parfid)
    sigma_fs8 = sigma_bb[Npar*z+Npop]
    sigma_apar = sigma_bb[Npar*z+Npop+1]
    sigma_aper = sigma_bb[Npar*z+Npop+2]
    print("{0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(zz,100*sigma_fs8/fs8fid,100*sigma_apar,100*sigma_aper))
    out_file.write("{0:.2f} {1:.2f} {2:.2f} {3:.2f}\n".format(zz,100*sigma_fs8/fs8fid,100*sigma_apar,100*sigma_aper))
    
# Loop over redshifts and translate [f, apar, aper] errors into [w0,wa,Om,Ok,h] errors
for z in range(Nz):
    zz = red[z]
    f1 = f(zz,parfid)*G(zz,parfid)
    apar1 = 1
    aper1 = 1
    for p in range(5):
        par2 = copy(parfid)
        if parfid[p] == 0:
            dpar = 0.01
        else:
            dpar = parfid[p]/100
        par2[p] += dpar
        f2 = f(zz,par2)*G(zz,par2)
        dbdw[p,z*Npar+Npop] = (f2 - f1)/dpar
        apar2 = H(zz,parfid)/H(zz,par2)
        dbdw[p,z*Npar+Npop+1] = (apar2 - apar1)/dpar
        aper2 = DA(zz,par2)/DA(zz,parfid)
        dbdw[p,z*Npar+Npop+2] = (aper2 - aper1)/dpar

Fpp = dot(dbdw,dot(Fbb,transpose(dbdw)))
Cpp = inv(Fpp)
sigma_pp = sqrt(diag(Cpp))

# The constraints
print("w0 = {0:.3f} +- {1:.3f}".format(w0,sigma_pp[0]))
out_file.write("w0 = {0:.3f} +- {1:.3f}\n".format(w0,sigma_pp[0]))
print("wa = {0:.3f} +- {1:.3f}".format(wa,sigma_pp[1]))
out_file.write("wa = {0:.3f} +- {1:.3f}\n".format(wa,sigma_pp[1]))
print("Om = {0:.3f} +- {1:.3f}".format(Om,sigma_pp[2]))
out_file.write("Om = {0:.3f} +- {1:.3f}\n".format(Om,sigma_pp[2]))
print("Ok = {0:.3f} +- {1:.3f}".format(Ok,sigma_pp[3]))
out_file.write("Ok = {0:.3f} +- {1:.3f}\n".format(Ok,sigma_pp[3]))
print("H0 = {0:.3f} +- {1:.3f}".format(100*h,100*sigma_pp[4]))
out_file.write("H0 = {0:.3f} +- {1:.3f}\n".format(100*h,100*sigma_pp[4]))

# Figure of Merit
FOM = 1/sqrt(det(Cpp[0:1,0:1]))
print("FOM = {0:.2f}".format(FOM))
out_file.write("FOM = {0:.2f}\n".format(FOM))
out_file.close()


# In[ ]:



