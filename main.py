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

# Import functions from the utility_functions.py
from utility_functions import flow_reconstruction_lassoCV

# Specify the aneurysm to process.
ANY = 'ICA' # 'BT' for Basilar tip aneurysm, and 'ICA' for Internal carotid artery aneurysm

print('processing aneurysm: ', ANY)

# Directories for loading data and saving results
dir0 = '/data_depository/' # Put the directory where the data is stored
dir0 = dir0 + ANY + '/'
loaddir_geometry = dir0 + 'geometry/'
loaddir_CFD = dir0 + 'CFD_frames/'
loaddir_PTV = dir0 + 'PTV_frames/'
loaddir_MRI = dir0 + 'MRI_frames/'
savedir_result = dir0 + 'MRI_reconstructed/'
if not os.path.exists(savedir_result):
  os.makedirs(savedir_result)


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
# print(Nt_cfd, Nt_ptv)
Ny_LB, Nx_LB, Nz_LB = np.shape(Xn_LB)

# Construct the measurement matrix.
print('Construct measurement matrix')
j_MRI, i_MRI, k_MRI = np.where(mask_MRI==True)
Npts_MRI = len(j_MRI)
x_MRI = Xn_MRI[j_MRI, i_MRI, k_MRI]
y_MRI = Yn_MRI[j_MRI, i_MRI, k_MRI]
z_MRI = Zn_MRI[j_MRI, i_MRI, k_MRI]
Coor_MRI = np.vstack((x_MRI,y_MRI,z_MRI)).T
index_MRI = -np.ones(mask_MRI.shape).astype('int')
index_MRI[j_MRI, i_MRI, k_MRI] = range(Npts_MRI)
dx_MRI = Xn_MRI[1,1,1] - Xn_MRI[0,0,0]
dy_MRI = Yn_MRI[1,1,1] - Yn_MRI[0,0,0]
dz_MRI = Zn_MRI[1,1,1] - Zn_MRI[0,0,0]

j_LB, i_LB, k_LB = np.where(mask_LB==True)
Npts_LB = len(j_LB)
x_LB = Xn_LB[j_LB, i_LB, k_LB]
y_LB = Yn_LB[j_LB, i_LB, k_LB]
z_LB = Zn_LB[j_LB, i_LB, k_LB]
Coor_LB = np.vstack((x_LB,y_LB,z_LB)).T
index_LB = -np.ones(mask_LB.shape).astype('int')
index_LB[j_LB, i_LB, k_LB] = range(Npts_LB)
xn_LB = Xn_LB[0,:,0]
yn_LB = Yn_LB[:,0,0]
zn_LB = Zn_LB[0,0,:]

measurement_M = scysparse.lil_matrix((Npts_MRI, Npts_LB), dtype=np.float)
search_x = dx_MRI * 1
search_y = dy_MRI * 1
search_z = dz_MRI * 1
for ct_pt in range(Npts_MRI):
  x_center, y_center, z_center = x_MRI[ct_pt], y_MRI[ct_pt], z_MRI[ct_pt]
  # Find the points within the volume
  x_dist = x_LB - x_center
  y_dist = y_LB - y_center
  z_dist = z_LB - z_center
  loc = np.where((np.abs(x_dist) < search_x)*(np.abs(y_dist) < search_y)*(np.abs(z_dist) < search_z))[0]
  sinc_weights = np.sinc(x_dist[loc]/dx_MRI)*np.sinc(y_dist[loc]/dy_MRI)*np.sinc(z_dist[loc]/dz_MRI)
  # Find a block of points within the volume to determine the sum of the sinc weights
  i_min = np.min(np.where((xn_LB>x_center-search_x)*(xn_LB<x_center+search_x)))
  i_max = np.max(np.where((xn_LB>x_center-search_x)*(xn_LB<x_center+search_x)))
  j_min = np.min(np.where((yn_LB>y_center-search_y)*(yn_LB<y_center+search_y)))
  j_max = np.max(np.where((yn_LB>y_center-search_y)*(yn_LB<y_center+search_y)))
  k_min = np.min(np.where((zn_LB>z_center-search_z)*(zn_LB<z_center+search_z)))
  k_max = np.max(np.where((zn_LB>z_center-search_z)*(zn_LB<z_center+search_z)))
  Npts_intravoxel = (i_max - i_min + 1)*(j_max - j_min + 1)*(k_max - k_min + 1)
  x_dist_block = (Xn_LB[j_min:j_max+1,i_min:i_max+1,k_min:k_max+1] - x_center).flatten()
  y_dist_block = (Yn_LB[j_min:j_max+1,i_min:i_max+1,k_min:k_max+1] - y_center).flatten()
  z_dist_block = (Zn_LB[j_min:j_max+1,i_min:i_max+1,k_min:k_max+1] - z_center).flatten()
  mask_block = mask_LB[j_min:j_max+1,i_min:i_max+1,k_min:k_max+1]
  mag_block = np.ones(mask_block.shape)
  mag_block[mask_block==False] = mag_outside
  sinc_weights_total = np.sum(np.sinc(x_dist_block/dx_MRI)*np.sinc(y_dist_block/dy_MRI)*np.sinc(z_dist_block/dz_MRI)*mag_block.flatten())
  if Npts_intravoxel < 1:
    print(Npts_intravoxel)
  else:
    measurement_M[ct_pt, index_LB[j_LB[loc], i_LB[loc], k_LB[loc]]] = sinc_weights / sinc_weights_total
# Convert to csr  
measurement_M = measurement_M.tocsr()

# Save the measurement matrix.
scysparse.save_npz(savedir_result + 'measurement_M.npz', measurement_M)


# Load the CFD and PTV velocity frames as library components.
N_lb_cfd = 50
N_lb_ptv = 50
t_load_cfd = np.random.choice(Nt_cfd, N_lb_cfd, replace=False)
t_load_ptv = np.random.choice(Nt_ptv, N_lb_ptv, replace=False)
t_load_cfd.sort()
t_load_ptv.sort()
 
U_lb_cfd = np.zeros((Npts_LB, N_lb_cfd))
V_lb_cfd = np.zeros((Npts_LB, N_lb_cfd))
W_lb_cfd = np.zeros((Npts_LB, N_lb_cfd))
for ct,ct_load in enumerate(t_load_cfd):
  cfd_data = loadmat(loaddir_CFD + 'velocity_'+str(ct_load).zfill(3)+'.mat',squeeze_me=True)
  U_lb_cfd[:,ct] = cfd_data['Un'][j_LB,i_LB,k_LB]
  V_lb_cfd[:,ct] = cfd_data['Vn'][j_LB,i_LB,k_LB]
  W_lb_cfd[:,ct] = cfd_data['Wn'][j_LB,i_LB,k_LB]

U_lb_ptv = np.zeros((Npts_LB, N_lb_ptv))
V_lb_ptv = np.zeros((Npts_LB, N_lb_ptv))
W_lb_ptv = np.zeros((Npts_LB, N_lb_ptv))
for ct,ct_load in enumerate(t_load_ptv):
  ptv_data = loadmat(loaddir_PTV + 'velocity_'+str(ct_load).zfill(3)+'.mat',squeeze_me=True)
  U_lb_ptv[:,ct] = ptv_data['Un'][j_LB,i_LB,k_LB]
  V_lb_ptv[:,ct] = ptv_data['Vn'][j_LB,i_LB,k_LB]
  W_lb_ptv[:,ct] = ptv_data['Wn'][j_LB,i_LB,k_LB]

U_LB = np.hstack((U_lb_ptv, U_lb_cfd))
V_LB = np.hstack((V_lb_ptv, V_lb_cfd))
W_LB = np.hstack((W_lb_ptv, W_lb_cfd))


# Define the kernel locations and the sigma of the Gaussian function.
N_skip = 2
Xn_kn = Xn_MRI[::N_skip,::N_skip,::N_skip]
Yn_kn = Yn_MRI[::N_skip,::N_skip,::N_skip]
Zn_kn = Zn_MRI[::N_skip,::N_skip,::N_skip]
mask_kn = mask_MRI[::N_skip,::N_skip,::N_skip]
j_kn, i_kn, k_kn = np.where(mask_kn==True)
Npts_kn = len(j_kn)
x_kn = Xn_kn[j_kn, i_kn, k_kn]
y_kn = Yn_kn[j_kn, i_kn, k_kn]
z_kn = Zn_kn[j_kn, i_kn, k_kn]
Coor_kn = np.vstack((x_kn,y_kn,z_kn)).T
dx_kn = Xn_kn[1,1,1] - Xn_kn[0,0,0]
dy_kn = Yn_kn[1,1,1] - Yn_kn[0,0,0]
dz_kn = Zn_kn[1,1,1] - Zn_kn[0,0,0]
sigma_kn = np.ones(Npts_kn) * np.max([dx_kn, dy_kn, dz_kn])
# save the basic info of the flow reconstruction
rgr_info = {}
rgr_info['t_load_cfd'] = t_load_cfd
rgr_info['t_load_ptv'] = t_load_ptv
rgr_info['Xn_kn'] = Xn_kn
rgr_info['Yn_kn'] = Yn_kn
rgr_info['Zn_kn'] = Zn_kn
rgr_info['mask_kn'] = mask_kn
rgr_info['sigma_kn'] = sigma_kn
savemat(savedir_result + 'rgr_info.mat', rgr_info, do_compression=True)


# Load the MRI data.
U_mri = np.zeros((Npts_MRI, Nt_mri))
V_mri = np.zeros((Npts_MRI, Nt_mri))
W_mri = np.zeros((Npts_MRI, Nt_mri))
for ct in range(Nt_mri):
  mri_data = loadmat(loaddir_MRI + 'MRI_velocity_'+str(ct).zfill(3)+'.mat', squeeze_me=True)
  U_mri[:,ct] = mri_data['Un'][j_MRI, i_MRI, k_MRI]
  V_mri[:,ct] = mri_data['Vn'][j_MRI, i_MRI, k_MRI]
  W_mri[:,ct] = mri_data['Wn'][j_MRI, i_MRI, k_MRI]


# Perform the reconstruction and save the results.
for ct in range(Nt_mri):
  print('For ct = '+str(ct) + '/'+str(Nt_mri))
  results = {}
  results['Un_rec'] = np.zeros((Ny_LB, Nx_LB, Nz_LB))
  results['Un_rec_mri'] = np.zeros((Ny_MRI, Nx_MRI, Nz_MRI))
  results['Vn_rec'] = np.zeros((Ny_LB, Nx_LB, Nz_LB))
  results['Vn_rec_mri'] = np.zeros((Ny_MRI, Nx_MRI, Nz_MRI))
  results['Wn_rec'] = np.zeros((Ny_LB, Nx_LB, Nz_LB))
  results['Wn_rec_mri'] = np.zeros((Ny_MRI, Nx_MRI, Nz_MRI))

  # For U velocity
  val_rec, val_rec_m, rgr_coef, alpha_cv = \
    flow_reconstruction_lassoCV(Coor_MRI, U_mri[:,ct], measurement_M, U_LB, Coor_LB, Coor_kn=Coor_kn, sigma_kn=sigma_kn, cv=5, n_jobs=4, weighted=True)
  results['Un_rec'][j_LB,i_LB,k_LB] = val_rec
  results['Un_rec_mri'][j_MRI,i_MRI,k_MRI] = val_rec_m
  results['rgr_coef_u'] = rgr_coef
  results['alpha_cv_u'] = alpha_cv
  # Rescale the resulting velocity field based on the mean absolute value of the velocity.
  mean_abs_U_mri = np.mean(np.abs(U_mri[:,ct]))
  mean_abs_U_rgr_mri = np.mean(np.abs(val_rec_m))
  results['Un_rec_rescale'] = results['Un_rec'] * (mean_abs_U_mri/mean_abs_U_rgr_mri)
  results['Un_rec_mri_rescale'] = results['Un_rec_mri'] * (mean_abs_U_mri/mean_abs_U_rgr_mri)

  # For V velocity
  val_rec, val_rec_m, rgr_coef, alpha_cv = \
    flow_reconstruction_lassoCV(Coor_MRI, V_mri[:,ct], measurement_M, V_LB, Coor_LB, Coor_kn=Coor_kn, sigma_kn=sigma_kn, cv=5, n_jobs=4, weighted=True)
  results['Vn_rec'][j_LB,i_LB,k_LB] = val_rec
  results['Vn_rec_mri'][j_MRI,i_MRI,k_MRI] = val_rec_m
  results['rgr_coef_v'] = rgr_coef
  results['alpha_cv_v'] = alpha_cv
  # Rescale the resulting velocity field based on the mean absolute value of the velocity.
  mean_abs_V_mri = np.mean(np.abs(V_mri[:,ct]))
  mean_abs_V_rgr_mri = np.mean(np.abs(val_rec_m))
  results['Vn_rec_rescale'] = results['Vn_rec'] * (mean_abs_V_mri/mean_abs_V_rgr_mri)
  results['Vn_rec_mri_rescale'] = results['Vn_rec_mri'] * (mean_abs_V_mri/mean_abs_V_rgr_mri)

  # For w velocity
  val_rec, val_rec_m, rgr_coef, alpha_cv = \
    flow_reconstruction_lassoCV(Coor_MRI, W_mri[:,ct], measurement_M, W_LB, Coor_LB, Coor_kn=Coor_kn, sigma_kn=sigma_kn, cv=5, n_jobs=4, weighted=True)
  results['Wn_rec'][j_LB,i_LB,k_LB] = val_rec
  results['Wn_rec_mri'][j_MRI,i_MRI,k_MRI] = val_rec_m
  results['rgr_coef_w'] = rgr_coef
  results['alpha_cv_w'] = alpha_cv
  # Rescale the resulting velocity field based on the mean absolute value of the velocity.
  mean_abs_W_mri = np.mean(np.abs(W_mri[:,ct]))
  mean_abs_W_rgr_mri = np.mean(np.abs(val_rec_m))
  results['Wn_rec_rescale'] = results['Wn_rec'] * (mean_abs_W_mri/mean_abs_W_rgr_mri)
  results['Wn_rec_mri_rescale'] = results['Wn_rec_mri'] * (mean_abs_W_mri/mean_abs_W_rgr_mri)
  
  savemat(savedir_result + 'rec_lassoCV_'+str(ct).zfill(3)+'.mat', results, do_compression=True)





# Load the reconstructed results and analyze.
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Times']})
rc('text', usetex=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
j_MRI, i_MRI, k_MRI = np.where(mask_MRI==True)
Npts_MRI = len(j_MRI)

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
j_LB, i_LB, k_LB = np.where(mask_LB==True)
Npts_LB = len(j_LB)

# Load the MRI data.
U_mri = np.zeros((Npts_MRI, Nt_mri))
V_mri = np.zeros((Npts_MRI, Nt_mri))
W_mri = np.zeros((Npts_MRI, Nt_mri))
for ct in range(Nt_mri):
  mri_data = loadmat(loaddir_MRI + 'MRI_velocity_'+str(ct).zfill(3)+'.mat', squeeze_me=True)
  U_mri[:,ct] = mri_data['Un'][j_MRI, i_MRI, k_MRI]
  V_mri[:,ct] = mri_data['Vn'][j_MRI, i_MRI, k_MRI]
  W_mri[:,ct] = mri_data['Wn'][j_MRI, i_MRI, k_MRI]

# Load the resuts and do analysis.
rgr_info = loadmat(savedir_result + 'rgr_info.mat', squeeze_me=True)
mask_kn = rgr_info['mask_kn'] 
Npts_kn = np.sum(mask_kn.astype('int'))

U_rec = np.zeros((Npts_LB, Nt_mri))
V_rec = np.zeros((Npts_LB, Nt_mri))
W_rec = np.zeros((Npts_LB, Nt_mri))
for ct in range(Nt_mri):
  rgr_result = loadmat(savedir_result + 'rec_lassoCV_'+str(ct).zfill(3)+'.mat', squeeze_me=True)
  U_rec[:,ct] = rgr_result['Un_rec_rescale'][j_LB,i_LB,k_LB]
  V_rec[:,ct] = rgr_result['Vn_rec_rescale'][j_LB,i_LB,k_LB]
  W_rec[:,ct] = rgr_result['Wn_rec_rescale'][j_LB,i_LB,k_LB]

vel_mag_mri = (U_mri**2 + V_mri**2 + W_mri**2)**0.5
vel_mag_rec = (U_rec**2 + V_rec**2 + W_rec**2)**0.5
quaritles_list = [25,50,75]
quartiles_vel_mag_mri = np.zeros((3,Nt_mri))
quartiles_vel_mag_rec = np.zeros((3,Nt_mri))
for ct in range(Nt_mri):
  for ct_quartile in range(3):
    quartile = quaritles_list[ct_quartile]
    quartiles_vel_mag_mri[ct_quartile, ct] = np.percentile(vel_mag_mri[:,ct], quartile)
    quartiles_vel_mag_rec[ct_quartile, ct] = np.percentile(vel_mag_rec[:,ct], quartile)


fig1 = plt.figure(1, figsize=(4,3))    
plt.figure(1)
fontsize = 8
t_list = np.arange(Nt_mri).astype('float') / Nt_mri
ax = plt.subplot(1,1,1)
plt.plot(t_list, quartiles_vel_mag_mri[1], '^r-', lw=1.0, markersize=2, label='MRI')
plt.fill_between(t_list, quartiles_vel_mag_mri[0], quartiles_vel_mag_mri[2], color='r', alpha=0.4)
plt.plot(t_list, quartiles_vel_mag_rec[1], 'og-', lw=1.0, markersize=2, label='MSR')
plt.fill_between(t_list, quartiles_vel_mag_rec[0], quartiles_vel_mag_rec[2], color='g', alpha=0.4)
plt.ylim(bottom=0)
plt.xlim(t_list[0],t_list[-1])
plt.xlabel('t/T',fontsize=fontsize-1)
plt.ylabel('velocity magnitude (m/s)',fontsize=fontsize-1)
plt.legend(loc = 'upper left', fontsize=fontsize-1, frameon=False)
plt.tick_params(axis='both',labelsize=fontsize-1)
plt.tight_layout()
plt.savefig(savedir_result + 'velocity_mag_quartiles', dpi=600)
plt.clf()












