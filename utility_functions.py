# Import the python libraries
import os
import numpy as np
import sys
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
from scipy.ndimage.morphology import binary_dilation
from sklearn.linear_model import LassoCV
from scipy.interpolate import Rbf

# Import functions for evluating hemodynamic parameters
from EvaluatePressureGradient import NavierStokesMomentum
from UncertaintyAnalysisPressure import PressureUncertaintyEvaluation
from ErrorEstimationFunctions import pressure_gradient_error_estimation_from_divergence
from ErrorEstimationFunctions import weighted_standard_deviation_3D
from NumericalDifference import first_derivative_SOC


def flow_reconstruction_lassoCV(Coor_m, var_m, M_matrix, Lib_modes, Coor_lib, Coor_kn=None, sigma_kn=None, n_jobs=1, cv=5, weighted=True, positive=False):
  # This function performs the lasso regression with CV for determining alpha.
  # Inputs:
  # Coor_m: the xyz coordinates of the measurement data, with a shape of (Npts_m, 3).
  # var_m: the measured data (1d array with the length of Npts_m)
  # M_matrix: the sparse csr matrix which projects the library modes to the measurement locations, shape (Npts_m, Npts_lb)
  # Lib_modes; the library modes matrix, with the shape (Npts_lb, N_modes)
  # Coor_lib: the coordinates of the library data, with a shape of (Npts_lb, 3).
  # Coor_kn: the coordinates of the kernel center locations, with the shape of (Npts_kn, 3). If None (not given), a global reconstruction is perfromed.
  # sigma_kn: the 1 sigma of the Gaussian kernel, 1d array with the length of Npts_kn.
  # n_jobs: the number of CPUs to use during the cross validation
  # cv: the number of folds for cross validation for lassoCV
  # weighted: if True, each kernel will be optimized by minimizing the weighted residual, otherwise not weighted.
  # positive: if True, the regression coefficients will be positive only.
  # Outputs:
  # val_rec: the reconstructed flow field, 1d array of length Npts_lb
  # val_rec_m: the reconstructed field projected back to the measurement grid.
  # rgr_coef: the coefficients from the lassoCV.  If the localized approach is employed, the rgr_coef contains the coefficients for each kernel, with a shape of (N_modes, Npts_kn).
  # alpha_cv: the alpha value determiend from cross-validation, 1d array of length Npts_kn
  
  if Coor_kn is None:
    # Perform the global reconstruction.
    n_sample = len(var_m)
    rgr_lassoCV = LassoCV(cv=cv,n_jobs=n_jobs,positive=positive)
    # Perform the fitting.
    rgr_lassoCV.fit(np.matmul(M_matrix.A,Lib_modes), var_m)
    # Get the coefficients.
    rgr_coef = rgr_lassoCV.coef_
    # Get the reconstruction.
    val_rec = Lib_modes.dot(rgr_coef)
    # for the alpha
    alpha_cv = rgr_lassoCV.alpha_

  else:
    # Perform the local reconstruction for each kernel.
    Npts_lb, N_modes = np.shape(Lib_modes)
    N_kn = len(sigma_kn)
    
    # Prepare array to store the results.
    rgr_coef = np.zeros((N_modes, N_kn))
    kernel_coef_stack = np.zeros((Npts_lb, N_kn))
    val_rec_stack = np.zeros((Npts_lb, N_kn))
    alpha_cv = np.zeros(N_kn)

    # Loop through the kernels.
    for ct_kn in range(N_kn):
      if ct_kn % 10 ==0:
        print('  on kernel #'+str(ct_kn)+'/'+str(N_kn))
      x_center = Coor_kn[ct_kn,0]
      y_center = Coor_kn[ct_kn,1]
      z_center = Coor_kn[ct_kn,2]
      sigma = sigma_kn[ct_kn]

      # Find the distances between each library node location to the kernel center.
      x_lib = Coor_lib[:,0]
      y_lib = Coor_lib[:,1]
      z_lib = Coor_lib[:,2]
      r_lib = ((x_lib - x_center)**2 + (y_lib - y_center)**2 + (z_lib - z_center)**2)**0.5
      # Generate the matrix Phi that describes the kernel strength using a Gassian function.
      kernel_coef = np.exp(-r_lib**2 / sigma**2)
      kernel_coef[kernel_coef < np.exp(-4)] = 0.0
      kernel_coef_stack[:,ct_kn] = kernel_coef

      # Construct the linear operator for the kenel.
      x_m = Coor_m[:,0]
      y_m = Coor_m[:,1]
      z_m = Coor_m[:,2]
      r_m = ((x_m - x_center)**2 + (y_m - y_center)**2 + (z_m - z_center)**2)**0.5
      kernel_coef_m = np.exp(-r_m**2 / sigma**2)
      kernel_coef_m[kernel_coef_m < np.exp(-4)] = 0.0
      index_nonzero_m = np.where(kernel_coef_m > 0)[0]
      Npts_index_nonzero_m = len(index_nonzero_m)
      kernel_coef_m_op = np.diag(kernel_coef_m[index_nonzero_m])
      
      # The operator combined kernel strength, measurement matrix, library modes
      LHS_operator = np.matmul(M_matrix.A[index_nonzero_m,:], Lib_modes)

      rgr_lassoCV = LassoCV(cv=cv,n_jobs=n_jobs,positive=positive)
      # Perform the fitting.
      if weighted == True: # Minimize the weighted residual
        rgr_lassoCV.fit(np.matmul(kernel_coef_m_op, LHS_operator), kernel_coef_m_op.dot(var_m[index_nonzero_m]))
      else: 
        rgr_lassoCV.fit(LHS_operator, var_m[index_nonzero_m])
      # Get the coefficients.
      rgr_coef[:,ct_kn] = rgr_lassoCV.coef_
      # Get the reconstruction.
      val_rec_vect = Lib_modes.dot(rgr_lassoCV.coef_)
      val_rec_vect[kernel_coef==0] = np.nan
      val_rec_stack[:,ct_kn] = val_rec_vect
      # Get the alpha from the cross validation
      alpha_cv[ct_kn] = rgr_lassoCV.alpha_
    
    # Obtain the final reconstruction by weighted combination.
    sum_kernel_stack = np.sum(kernel_coef_stack, axis=1)
    sum_kernel_stack[sum_kernel_stack<1e-9] = 1.0
    val_rec = np.nansum(val_rec_stack * kernel_coef_stack, axis=1) / sum_kernel_stack
    
  # Also evaluated the measurement data based on the reconstructed field
  val_rec_m = M_matrix.dot(val_rec)

  # return the results.
  return val_rec,  val_rec_m, rgr_coef, alpha_cv


def mask_truncation_3d(mask_raw,minimum_dim=3):
  # trancate the fluid_mask.
  # The minimum dim indicate the minimum number of connecting points 
  # Due to the finite difference scheme, the node should be in part of a 3x3x3 cube to get correct second order derivative.
  # Inputs: 
  #   mask_raw: the original flow mask
  #   minmum_dim: the minimum number of connecting points 
  # Returns:
  #   fluid_mask_truncate: the truncated flow mask
  
  Ny,Nx,Nz = np.shape(mask_raw)
  md = minimum_dim
  # Check the cube in the domain one by one.
  fluid_index = np.zeros(mask_raw.shape).astype('int')
  for j in range(Ny-md+1):
    for i in range(Nx-md+1):
      for k in range(Nz-md+1):
        cube = mask_raw[j:j+md,i:i+md,k:k+md]
        j_c,i_c,k_c = np.where(cube==False)
        if len(j_c) == 0:
          fluid_index[j:j+md,i:i+md,k:k+md] += np.ones((md,md,md)).astype('int')
  fluid_mask_truncate = np.copy(mask_raw)
  fluid_mask_truncate[np.where(fluid_index==0)] = False
  
  return fluid_mask_truncate


def WSS_calculation_fast(x_wall, y_wall, z_wall, normal_x, normal_y, normal_z, x,y,z,u,v,w, mu, window_r, N_inner=10, N_wall=10, min_dist=0.1e-3):
  # This function evaluates the WSS at given wall locations.
  # The gradients are evaluated by thin-plate interpolation.
  # For each wall point, the neighboring N_wall wall points and neighboring N_inner inner points are used.
  # Inputs:
  # x_wall, y_wall, z_wall: the wall locations, 1d arrays
  # normal_x, normal_y, normal_z: the wall normals, 1d arrays.
  # x,y,z: the location of the velocity values (inner)
  # u,v,w: the velocity values at xyz locations. from a single snapshot.
  # mu: dynamic viscosity
  # window_r: the maximum search radius, the points within the window is used for rbf interpolation.
  # N_innner, N_wall: the maximum number of inner and wall points allowed for the rbf interpolation.
  # min_dist: the minimum distance allowed from the wall to determine the velocity gradient (to avoid singularity)
  # Outputs: 
  # tau_mag, tau_x, tau_y, tau_z: the 1d arrays of magnitude and components of WSS vector

  Npts_wall = len(x_wall)
  Npts_vel = len(x)
  print('Npts_wall, Npts_vel: ', Npts_wall, Npts_vel)
  h = 1e-6

  u_wall = np.zeros(Npts_wall)
  v_wall = np.zeros(Npts_wall)
  w_wall = np.zeros(Npts_wall)

  # Remove the inner points that is too close to the wall points.
  dist_min = np.zeros(Npts_vel)
  for ct_pt in range(Npts_vel):
    dist_min[ct_pt] = np.min(((x[ct_pt] - x_wall)**2 + (y[ct_pt] - y_wall)**2 + (z[ct_pt] - z_wall)**2)**0.5)
  arg_keep = np.where(dist_min >= min_dist)[0]
  x = x[arg_keep]
  y = y[arg_keep]
  z = z[arg_keep]
  u = u[arg_keep]
  v = v[arg_keep]
  w = w[arg_keep]
  Npts_vel = len(x)
  print('Npts_vel: ',Npts_vel)
  
  tau_x = np.zeros(Npts_wall)
  tau_y = np.zeros(Npts_wall)
  tau_z = np.zeros(Npts_wall)

  for ct_pt in range(Npts_wall):

    ct_disp_list = np.linspace(0,Npts_wall-1,10).astype('int')
    if ct_pt in ct_disp_list:
      print('working on ct = '+str(ct_pt)+'/'+str(Npts_wall))

    x_pt = x_wall[ct_pt]
    y_pt = y_wall[ct_pt]
    z_pt = z_wall[ct_pt]

    # Find the wall points and vel points for the interpolation.
    wall_dist = ((x_wall - x_pt)**2 + (y_wall - y_pt)**2 + (z_wall - z_pt)**2)**0.5
    N_use = np.min([N_wall, (wall_dist<=window_r).astype('int').sum()])
    ind_use_wall = np.argsort(wall_dist)[:N_use]

    inner_dist = ((x - x_pt)**2 + (y - y_pt)**2 + (z - z_pt)**2)**0.5
    N_use = np.min([N_inner, (inner_dist<=window_r).astype('int').sum()])
    ind_use_inner = np.argsort(inner_dist)[:N_use]

    x_list = np.concatenate((x_wall[ind_use_wall], x[ind_use_inner]))
    y_list = np.concatenate((y_wall[ind_use_wall], y[ind_use_inner]))
    z_list = np.concatenate((z_wall[ind_use_wall], z[ind_use_inner]))
    u_list = np.concatenate((u_wall[ind_use_wall], u[ind_use_inner]))
    v_list = np.concatenate((v_wall[ind_use_wall], v[ind_use_inner]))
    w_list = np.concatenate((w_wall[ind_use_wall], w[ind_use_inner]))

    rbfi = Rbf(x_list, y_list, z_list, u_list, function='thin_plate')
    du_dx = (rbfi(x_pt+h, y_pt, z_pt) - rbfi(x_pt-h, y_pt, z_pt)) / (2.0*h)
    du_dy = (rbfi(x_pt, y_pt+h, z_pt) - rbfi(x_pt, y_pt-h, z_pt)) / (2.0*h)
    du_dz = (rbfi(x_pt, y_pt, z_pt+h) - rbfi(x_pt, y_pt, z_pt-h)) / (2.0*h)

    rbfi = Rbf(x_list, y_list, z_list, v_list, function='thin_plate')
    dv_dx = (rbfi(x_pt+h, y_pt, z_pt) - rbfi(x_pt-h, y_pt, z_pt)) / (2.0*h)
    dv_dy = (rbfi(x_pt, y_pt+h, z_pt) - rbfi(x_pt, y_pt-h, z_pt)) / (2.0*h)
    dv_dz = (rbfi(x_pt, y_pt, z_pt+h) - rbfi(x_pt, y_pt, z_pt-h)) / (2.0*h)

    rbfi = Rbf(x_list, y_list, z_list, w_list, function='thin_plate')
    dw_dx = (rbfi(x_pt+h, y_pt, z_pt) - rbfi(x_pt-h, y_pt, z_pt)) / (2.0*h)
    dw_dy = (rbfi(x_pt, y_pt+h, z_pt) - rbfi(x_pt, y_pt-h, z_pt)) / (2.0*h)
    dw_dz = (rbfi(x_pt, y_pt, z_pt+h) - rbfi(x_pt, y_pt, z_pt-h)) / (2.0*h)

    # Evaluate the taus
    nx = normal_x[ct_pt]
    ny = normal_y[ct_pt]
    nz = normal_z[ct_pt]
    nmag = (nx**2 + ny**2 + nz**2)**0.5
    tau_x[ct_pt] = mu*(2*nx*du_dx + ny*(du_dy + dv_dx) + nz*(du_dz + dw_dx)) / nmag
    tau_y[ct_pt] = mu*(nx*(du_dy + dv_dx) + 2*ny*dv_dy + nz*(dv_dz + dw_dy)) / nmag
    tau_z[ct_pt] = mu*(nx*(du_dz + dw_dx) + ny*(dv_dz + dw_dy) + 2*nz*dw_dz) / nmag

  tau_mag = (tau_x**2 + tau_y**2 + tau_z**2)**0.5

  return tau_mag, tau_x, tau_y, tau_z


def WLS_pressure_evaluation(Xn,Yn,Zn,fluid_mask,Un,Vn,Wn,nu,rho,dt):
  # This function performs the pressure reconstruction from the velocity fields using the measurement-error-based weighted least-squares approach.
  # For detials on the methodology, please refer to: Zhang, J., M. C. Brindise, S. Rothenberger, S. Schnell, M. Markl, D. Saloner, V. L. Rayz, and P. P. Vlachos. 4D Flow MRI Pressure Estimation Using Velocity Measurement-Error based Weighted Least-Squares. IEEE Trans. Med. Imaging 39:1668–1680, 2020.
  # All inputs and outputs in SI units
  # Inputs:
  # Xn, Yn, Zn: the 3D matrix for the grid coordinates, shape (Ny,Nx,Nz)
  # fluid_mask: binary mask in 3D for the velocity data, in shape (Ny,Nx,Nz)
  # Un, Vn, Wn: velocity fields, in shape (Nt,Ny,Nx,Nz)
  # nu, rho: the kinematic viscosity and density of the fluid.
  # dt: the time difference between time frames.
  # Outputs:
  # P_WLS_div_wstd: the reconstructed pressure fields, in shape (Nt,Ny,Nx,Nz)

  Nt,Ny,Nx,Nz = np.shape(Un)

  print('Calculating pgrad')
  pressure_gradient_calculator = NavierStokesMomentum(Xn,Yn,Zn,Un,Vn,Wn,fluid_mask,nu,rho,dt,gravity_direction=None)
  dP_dx_st, dP_dy_st, dP_dz_st, mask_dp_dx, mask_dp_dy, mask_dp_dz = pressure_gradient_calculator.eval_pressure_gradient_staggered()
  j_x, i_x, k_x = np.where(mask_dp_dx)
  Npts_x = len(j_x)
  j_y, i_y, k_y = np.where(mask_dp_dy)
  Npts_y = len(j_y)
  j_z, i_z, k_z = np.where(mask_dp_dz)
  Npts_z = len(j_z)
  pgrad_stack = np.zeros((Nt,Npts_x+Npts_y+Npts_z))
  for ct in range(Nt):
    pgrad_stack[ct,:Npts_x] = dP_dx_st[ct,j_x,i_x,k_x]
    pgrad_stack[ct,Npts_x:Npts_x+Npts_y] = dP_dy_st[ct,j_y,i_y,k_y]
    pgrad_stack[ct,Npts_x+Npts_y:] = dP_dz_st[ct,j_z,i_z,k_z]

  # Estimate the pressure gradient errors from the velocity divergence. 
  print('Estimating pgrad error from velocity divergence')
  dP_dx_st, dP_dy_st, dP_dz_st, dP_dx_st_error, dP_dy_st_error, dP_dz_st_error = \
    pressure_gradient_error_estimation_from_divergence(Xn,Yn,Zn,fluid_mask,Un,Vn,Wn,nu,rho,dt,gravity_direction=None)

  # Calculate the Weighted STD of the estimated pressure gradient errors.
  print('Calculating WSTD of dp/dx errors')
  dP_dx_st_error_wstd = weighted_standard_deviation_3D(dP_dx_st_error,mask_dp_dx,sigma_r=2,sigma_t=1)
  print('Calculating WSTD of dp/dy errors')
  dP_dy_st_error_wstd = weighted_standard_deviation_3D(dP_dy_st_error,mask_dp_dy,sigma_r=2,sigma_t=1)
  print('Calculating WSTD of dp/dz errors')
  dP_dz_st_error_wstd = weighted_standard_deviation_3D(dP_dz_st_error,mask_dp_dz,sigma_r=2,sigma_t=1)

  # Pressure reconstruction
  pressure_calculator = PressureUncertaintyEvaluation(Xn,Yn,Zn,Un,Vn,Wn,fluid_mask,nu,rho,dt)
  # For WLS pressure reconstruction (divergence based and wstd)
  print('Calculating P_WLS_div wstd')
  P_WLS_div_wstd = np.zeros((Nt,Ny,Nx,Nz))
  cov_pgrad_st_diag = np.zeros(Npts_x+Npts_y+Npts_z)
  cov_low_limit = 1e-3
  for ct in range(Nt):
    print('  at ct = '+str(ct))
    pgrad_st = pgrad_stack[ct]
    cov_pgrad_st_diag[:Npts_x] = dP_dx_st_error_wstd[ct,j_x,i_x,k_x]**2
    cov_pgrad_st_diag[Npts_x:Npts_x+Npts_y] = dP_dy_st_error_wstd[ct,j_y,i_y,k_y]**2
    cov_pgrad_st_diag[Npts_x+Npts_y:] = dP_dz_st_error_wstd[ct,j_z,i_z,k_z]**2
    cov_pgrad_st_diag[np.where(cov_pgrad_st_diag < cov_low_limit)] = cov_low_limit
    cov_pgrad_st = scysparse.diags(cov_pgrad_st_diag,format='csr',dtype=np.float)
    P_WLS_div_wstd[ct] = pressure_calculator.evaluate_pressure_weighted_least_square(pgrad_st,cov_pgrad_st,ref_point=None)

  return P_WLS_div_wstd


