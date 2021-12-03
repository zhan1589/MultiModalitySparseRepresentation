# Import python libraries and functions.
import os
import numpy as np
import sys
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
from scipy.io import loadmat, savemat
from scipy.spatial import Delaunay
from scipy.interpolate import griddata, RegularGridInterpolator, interpn
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from scipy.ndimage import measurements
from scipy.ndimage.measurements import label as ndimage_label

# Import functions from the utility_functions.py
from utility_functions import WLS_pressure_evaluation, WSS_calculation_fast, mask_truncation_3d


# Define the wrapper functions to calcualte and save the hemodynamic resutls.
def hemodynamic_evaluation_comnbined(Xn,Yn,Zn,Un,Vn,Wn,mask,x_wall,y_wall,z_wall,nx_wall,ny_wall,nz_wall,dt,mu,rho,savedir=None):
  # This function evaluate the pressure and wss
  # and save the results into the given directory.

  nu = mu/rho
  Nt,Ny,Nx,Nz = np.shape(Un)
  Npts_wall = len(x_wall)
  dx = Xn[1,1,1] - Xn[0,0,0]
  dy = Yn[1,1,1] - Yn[0,0,0]
  dz = Zn[1,1,1] - Zn[0,0,0]
  j,i,k = np.where(mask==True)
  
  print('Pressure evaluation')
  mask_truncated = mask_truncation_3d(mask,minimum_dim=3)
  # Only keep the largest region of mask.
  label, num_features = ndimage_label(mask_truncated)
  if num_features > 1:
    print('warning, num_features = ',num_features)
    print('Npts: ',np.sum(mask_truncated.astype('int')))
    feature_sizes = np.zeros(num_features)
    for ct_label in range(num_features):
      feature_sizes[ct_label] = (label == ct_label+1).astype('int').sum()
    largest_feature_label = np.argsort(feature_sizes)[-1] + 1
    mask_truncated = (label == largest_feature_label)
    print('Npts: ',np.sum(mask_truncated.astype('int')))

  P_WLS = WLS_pressure_evaluation(Xn,Yn,Zn,mask_truncated,Un,Vn,Wn,nu,rho,dt)
  pressure_result = {}
  pressure_result['mask_truncated'] = mask_truncated
  pressure_result['P_WLS'] = P_WLS
  savemat(savedir + 'pressure_result.mat', pressure_result, do_compression=True)

  print('WSS calcualtion')
  window_r = np.max([dx,dy,dz])*2
  tau_mag = np.zeros((Nt,Npts_wall))
  tau_x = np.zeros((Nt,Npts_wall))
  tau_y = np.zeros((Nt,Npts_wall))
  tau_z = np.zeros((Nt,Npts_wall))
  j,i,k = np.where(mask==True)
  x = Xn[j,i,k]
  y = Yn[j,i,k]
  z = Zn[j,i,k]
  for ct in range(Nt):
    u = Un[ct,j,i,k]
    v = Vn[ct,j,i,k]
    w = Wn[ct,j,i,k]
    tau_mag[ct], tau_x[ct], tau_y[ct], tau_z[ct] = \
      WSS_calculation_fast(x_wall, y_wall, z_wall, nx_wall, ny_wall, nz_wall, x,y,z,u,v,w, mu, window_r, N_inner=10, N_wall=10)

  TAWSS = np.mean(tau_mag,axis=0)
  OSI = 0.5*(1 - (np.mean(tau_x,axis=0)**2 + np.mean(tau_y,axis=0)**2 + np.mean(tau_z,axis=0)**2)**0.5/TAWSS)
  RRT = 1.0 / ((1 - 2*OSI)*TAWSS)
  wss_result = {}
  wss_result['x_wall'] = x_wall
  wss_result['y_wall'] = y_wall
  wss_result['z_wall'] = z_wall
  wss_result['nx_wall'] = nx_wall
  wss_result['ny_wall'] = ny_wall
  wss_result['nz_wall'] = nz_wall
  wss_result['tau_mag'] = tau_mag
  wss_result['TAWSS'] = TAWSS
  wss_result['OSI'] = OSI
  wss_result['RRT'] = RRT
  savemat(savedir + 'wss_result.mat', wss_result, do_compression=True)


###################################################################################

# Start of the main function.
# Specify the aneurysm to process.
ANY = 'BT' # 'BT' for Basilar tip aneurysm, and 'ICA' for Internal carotid artery aneurysm

print('processing aneurysm: ', ANY)

# Directories for loading data and saving results
dir0 = '/data_depository/' # Put the directory where the data is stored
dir0 = dir0 + ANY + '/'
loaddir_geometry = dir0 + 'geometry/'
loaddir_CFD = dir0 + 'CFD_frames/'
loaddir_PTV = dir0 + 'PTV_frames/'
loaddir_MRI = dir0 + 'MRI_frames/'
savedir_result = dir0 + 'MRI_reconstructed/'
savedir_hemodynamic = dir0 + 'Hemodynamic_results/'
if not os.path.exists(savedir_hemodynamic):
  os.makedirs(savedir_hemodynamic)


# Load the basic info for the processing.
basic_info = loadmat(dir0 + 'basic_info.mat', squeeze_me=True)
# fluid density and viscosity
rho = basic_info['rho']  
mu = basic_info['mu'] 
# time difference between frames.
dt_MRI = basic_info['dt_MRI'] 
# Coordinates of the pressure reference point.
x_ref = basic_info['x_ref']
y_ref = basic_info['y_ref'] 
z_ref = basic_info['z_ref'] 

# Load the grid information for MRI data.
grid_mri = loadmat(loaddir_geometry + 'MRI_grid.mat', squeeze_me=True)
Xn_MRI = grid_mri['Xn']
Yn_MRI = grid_mri['Yn']
Zn_MRI = grid_mri['Zn']
mask_MRI = grid_mri['mask'].astype('bool')
t_points_mri = grid_mri['t_points']
saturation_ratio = grid_mri['saturation_ratio']
mag_outside = 1.0/saturation_ratio
Nt_mri = len(t_points_mri)
Ny_MRI, Nx_MRI, Nz_MRI = np.shape(Xn_MRI)

# Load the grid information for the CFD and PTV frames for constructing library.
grid_lib = loadmat(loaddir_geometry + 'Library_grid.mat', squeeze_me=True)
Xn_LB = grid_lib['Xn'] 
Yn_LB = grid_lib['Yn'] 
Zn_LB = grid_lib['Zn'] 
mask_LB = grid_lib['mask'] 
t_points_cfd = grid_lib['t_points_cfd']
t_points_ptv = grid_lib['t_points_ptv'] 
Nt_cfd = len(t_points_cfd)
Nt_ptv = len(t_points_ptv)
Ny_LB, Nx_LB, Nz_LB = np.shape(Xn_LB)

# Load the wall points.
wall_info = loadmat(loaddir_geometry + 'wall_points.mat', squeeze_me=True)
x_wall = wall_info['x']
y_wall = wall_info['y']
z_wall = wall_info['z']
nx_wall = wall_info['nx']
ny_wall = wall_info['ny']
nz_wall = wall_info['nz']

# Load the MRI velocity data.
print('Load mri velocity')
Un_mri = np.zeros((Nt_mri, Ny_MRI, Nx_MRI, Nz_MRI))
Vn_mri = np.zeros((Nt_mri, Ny_MRI, Nx_MRI, Nz_MRI))
Wn_mri = np.zeros((Nt_mri, Ny_MRI, Nx_MRI, Nz_MRI))
for ct in range(Nt_mri):
  mri_data = loadmat(loaddir_MRI + 'MRI_velocity_'+str(ct).zfill(3)+'.mat', squeeze_me=True)
  Un_mri[ct] = mri_data['Un']
  Vn_mri[ct] = mri_data['Vn']
  Wn_mri[ct] = mri_data['Wn']

# Load the reconstructed velocity fields.
print('Load reconstruction results')
Un_rec = np.zeros((Nt_mri, Ny_LB, Nx_LB, Nz_LB))
Vn_rec = np.zeros((Nt_mri, Ny_LB, Nx_LB, Nz_LB))
Wn_rec = np.zeros((Nt_mri, Ny_LB, Nx_LB, Nz_LB))
for ct in range(Nt_mri):
  results = loadmat(savedir_result + 'rec_lassoCV_'+str(ct).zfill(3)+'.mat', squeeze_me=True)
  Un_rec[ct] = results['Un_rec_rescale']
  Vn_rec[ct] = results['Vn_rec_rescale']
  Wn_rec[ct] = results['Wn_rec_rescale']

print('hemodynamic evaluation mri')
savedir_hemo = savedir_hemodynamic + 'hemodynamic_results_mri/'
if not os.path.exists(savedir_hemo):
  os.makedirs(savedir_hemo)
hemodynamic_evaluation_comnbined(Xn_MRI,Yn_MRI,Zn_MRI,Un_mri,Vn_mri,Wn_mri,mask_MRI,x_wall,y_wall,z_wall,nx_wall,ny_wall,nz_wall,dt_MRI,mu,rho,savedir=savedir_hemo)

print('hemodynamic evaluation reconstructed fields')
savedir_hemo = savedir_hemodynamic + 'hemodynamic_results_rec/'
if not os.path.exists(savedir_hemo):
  os.makedirs(savedir_hemo)
hemodynamic_evaluation_comnbined(Xn_LB,Yn_LB,Zn_LB,Un_rec,Vn_rec,Wn_rec,mask_LB,x_wall,y_wall,z_wall,nx_wall,ny_wall,nz_wall,dt_MRI,mu,rho,savedir=savedir_hemo)




# Load the hemodynamic results and analyze.
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Times']})
rc('text', usetex=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the basic info for the processing.
basic_info = loadmat(dir0 + 'basic_info.mat', squeeze_me=True)
# Coordinates of the pressure reference point.
x_ref = basic_info['x_ref']
y_ref = basic_info['y_ref'] 
z_ref = basic_info['z_ref'] 

# Load the grid information for MRI data.
grid_mri = loadmat(loaddir_geometry + 'MRI_grid.mat', squeeze_me=True)
Xn_MRI = grid_mri['Xn']
Yn_MRI = grid_mri['Yn']
Zn_MRI = grid_mri['Zn']
mask_MRI = grid_mri['mask'].astype('bool')
t_points_mri = grid_mri['t_points']
saturation_ratio = grid_mri['saturation_ratio']
mag_outside = 1.0/saturation_ratio
Nt_mri = len(t_points_mri)
Ny_MRI, Nx_MRI, Nz_MRI = np.shape(Xn_MRI)
xn_MRI = Xn_MRI[0,:,0]
yn_MRI = Yn_MRI[:,0,0]
zn_MRI = Zn_MRI[0,0,:]

# Load the grid information for the CFD and PTV frames for constructing library.
grid_lib = loadmat(loaddir_geometry + 'Library_grid.mat', squeeze_me=True)
Xn_LB = grid_lib['Xn'] 
Yn_LB = grid_lib['Yn'] 
Zn_LB = grid_lib['Zn'] 
mask_LB = grid_lib['mask'] 
t_points_cfd = grid_lib['t_points_cfd']
t_points_ptv = grid_lib['t_points_ptv'] 
Nt_cfd = len(t_points_cfd)
Nt_ptv = len(t_points_ptv)
Ny_LB, Nx_LB, Nz_LB = np.shape(Xn_LB)
xn_LB = Xn_LB[0,:,0]
yn_LB = Yn_LB[:,0,0]
zn_LB = Zn_LB[0,0,:]

# Load the hemodynamic results.
savedir_hemo = savedir_hemodynamic + 'hemodynamic_results_mri/'
pressure_result = loadmat(savedir_hemo + 'pressure_result.mat', squeeze_me=True)
mask_MRI_truncated = pressure_result['mask_truncated'] 
P_MRI = pressure_result['P_WLS']
wss_result = loadmat(savedir_hemo + 'wss_result.mat', squeeze_me=True)
WSS_MRI = wss_result['tau_mag']

savedir_hemo = savedir_hemodynamic + 'hemodynamic_results_rec/'
pressure_result = loadmat(savedir_hemo + 'pressure_result.mat', squeeze_me=True)
mask_MSR_truncated = pressure_result['mask_truncated'] 
P_MSR = pressure_result['P_WLS']
wss_result = loadmat(savedir_hemo + 'wss_result.mat', squeeze_me=True)
WSS_MSR = wss_result['tau_mag']

Nt_mri, Npts_wall = np.shape(WSS_MRI)

# Adjust the pressure field by setting the pressure at the reference point to be zero (in order to compare between datasets)
i_LB_ref = np.argsort(np.abs(xn_LB - x_ref))[0]
j_LB_ref = np.argsort(np.abs(yn_LB - y_ref))[0]
k_LB_ref = np.argsort(np.abs(zn_LB - z_ref))[0]
i_MRI_ref = np.argsort(np.abs(xn_MRI - x_ref))[0]
j_MRI_ref = np.argsort(np.abs(yn_MRI - y_ref))[0]
k_MRI_ref = np.argsort(np.abs(zn_MRI - z_ref))[0]
for ct in range(Nt_mri):
  P_MRI[ct] -= P_MRI[ct, j_MRI_ref, i_MRI_ref, k_MRI_ref]
  P_MSR[ct] -= P_MSR[ct, j_LB_ref, i_LB_ref, k_LB_ref]

# Find the region of resolvable pressure for both datasets.
j_LB, i_LB, k_LB = np.where(mask_MSR_truncated==True)
mask_LB_core = np.zeros((Ny_LB, Nx_LB, Nz_LB))
points = (Xn_MRI[0,:,0], Yn_MRI[:,0,0], Zn_MRI[0,0,:])
values = np.transpose(mask_MRI_truncated.astype('float'), (1,0,2))
xi = np.vstack((Xn_LB[j_LB,i_LB,k_LB], Yn_LB[j_LB,i_LB,k_LB], Zn_LB[j_LB,i_LB,k_LB])).T
mask_LB_core[j_LB,i_LB,k_LB] = interpn(points, values, xi, method='linear', bounds_error=False, fill_value=0.0)
mask_LB_core = mask_LB_core > 0.5

# Plot the time series of pressure and wss distributions.
j_MRI, i_MRI, k_MRI = np.where(mask_MRI_truncated==True)
j_LB, i_LB, k_LB = np.where(mask_LB_core==True)
Npts_MRI = len(j_MRI)
Npts_LB = len(j_LB)
p_data_mri = np.zeros((Npts_MRI, Nt_mri))
p_data_msr = np.zeros((Npts_LB, Nt_mri))
for ct in range(Nt_mri):
  p_data_mri[:,ct] = P_MRI[ct,j_MRI, i_MRI, k_MRI]
  p_data_msr[:,ct] = P_MSR[ct,j_LB, i_LB, k_LB]

quaritles_list = [25,50,75]
quartiles_p_mri = np.zeros((3,Nt_mri))
quartiles_p_msr = np.zeros((3,Nt_mri))
quartiles_wss_mri = np.zeros((3,Nt_mri))
quartiles_wss_msr = np.zeros((3,Nt_mri))
for ct in range(Nt_mri):
  for ct_quartile in range(3):
    quartile = quaritles_list[ct_quartile]
    quartiles_p_mri[ct_quartile, ct] = np.percentile(p_data_mri[:,ct], quartile)
    quartiles_p_msr[ct_quartile, ct] = np.percentile(p_data_msr[:,ct], quartile)
    quartiles_wss_mri[ct_quartile, ct] = np.percentile(WSS_MRI[ct,:], quartile)
    quartiles_wss_msr[ct_quartile, ct] = np.percentile(WSS_MSR[ct,:], quartile)


fig1 = plt.figure(1, figsize=(7,3))    
plt.figure(1)
fontsize = 8
t_list = np.arange(Nt_mri).astype('float') / Nt_mri
ax = plt.subplot(1,2,1)
plt.plot(t_list, quartiles_p_mri[1], '^r-', lw=1.0, markersize=2, label='MRI')
plt.fill_between(t_list, quartiles_p_mri[0], quartiles_p_mri[2], color='r', alpha=0.4)
plt.plot(t_list, quartiles_p_msr[1], 'og-', lw=1.0, markersize=2, label='MSR')
plt.fill_between(t_list, quartiles_p_msr[0], quartiles_p_msr[2], color='g', alpha=0.4)
plt.xlim(t_list[0],t_list[-1])
plt.xlabel('t/T',fontsize=fontsize-1)
plt.ylabel('pressure (Pa)',fontsize=fontsize-1)
plt.legend(loc = 'upper left', fontsize=fontsize-1, frameon=False)
plt.tick_params(axis='both',labelsize=fontsize-1)
ax = plt.subplot(1,2,2)
plt.plot(t_list, quartiles_wss_mri[1], '^r-', lw=1.0, markersize=2, label='MRI')
plt.fill_between(t_list, quartiles_wss_mri[0], quartiles_wss_mri[2], color='r', alpha=0.4)
plt.plot(t_list, quartiles_wss_msr[1], 'og-', lw=1.0, markersize=2, label='MSR')
plt.fill_between(t_list, quartiles_wss_msr[0], quartiles_wss_msr[2], color='g', alpha=0.4)
plt.xlim(t_list[0],t_list[-1])
plt.ylim(bottom=0)
plt.xlabel('t/T',fontsize=fontsize-1)
plt.ylabel('WSS (Pa)',fontsize=fontsize-1)
plt.legend(loc = 'upper left', fontsize=fontsize-1, frameon=False)
plt.tick_params(axis='both',labelsize=fontsize-1)
plt.tight_layout()
plt.savefig(savedir_hemodynamic + 'p_wss_quartiles', dpi=600)
plt.clf()

















