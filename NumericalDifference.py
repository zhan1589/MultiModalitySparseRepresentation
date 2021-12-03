'''
Numerical differencing functions for 3D data fields.
Contains functions for 1st derivative, 2nd derivative, and divergence operation.
Second order central difference is employed
'''

import numpy as np
    
def first_derivative_SOC(U,h,axis=0,mask=None):
  """
  # Evaluate the first derivative of a 3d matrix using second order central difference scheme,
  # biased to 1st order at the boundaries.
  # The input variable U is the 3D matix containing the data
  # The input h is the grid spacing for the interested axis
  # axis specify the axis for evaluating
  # mask define the mask of the data. If mask is None, the mask is full True.
  Inputs:
    U: 3d field of the variable
    h: grid size. 
    axis: along which dimension the derivative is calculated.
    mask: binary maks of the field.
  Returns:
    dU_dy/dU_dx/dU_dz: 3d field of the derivative.
  """
  
  Ny,Nx,Nz = np.shape(U)
  invh = 1.0/h
  
  U_ex = np.zeros((Ny+2,Nx+2,Nz+2))
  U_ex[1:-1,1:-1,1:-1] = U
  U = np.copy(U_ex)
  
  mask_ex = np.zeros((Ny+2,Nx+2,Nz+2)).astype('bool')
  if mask is None:
    mask_ex[1:-1,1:-1,1:-1] = True
  else:
    mask_ex[1:-1,1:-1,1:-1] = mask
    
  if axis == 0:  # Evaluate the derivative along y axis
    dU_dy = np.zeros((Ny+2,Nx+2,Nz+2))
    dU_dy[1:-1,1:-1,1:-1] = (U[2:,1:-1,1:-1] - U[:-2,1:-1,1:-1])*0.5*invh
    # Deal with the boundary points, use 1st order biased
    jj_C,ii_C,kk_C = np.where(mask_ex==True)
    maskN = mask_ex[jj_C+1,ii_C,kk_C]
    maskS = mask_ex[jj_C-1,ii_C,kk_C]
    N_boundary = np.where(maskN==False)
    j,i,k = jj_C[N_boundary],ii_C[N_boundary],kk_C[N_boundary]
    dU_dy[j,i,k] = (U[j,i,k] - U[j-1,i,k])*invh
    S_boundary = np.where(maskS==False)
    j,i,k = jj_C[S_boundary],ii_C[S_boundary],kk_C[S_boundary]
    dU_dy[j,i,k] = (U[j+1,i,k] - U[j,i,k])*invh
    NS_boundary = np.where(np.logical_and(maskN==False,maskS==False))
    j,i,k = jj_C[NS_boundary],ii_C[NS_boundary],kk_C[NS_boundary]
    dU_dy[j,i,k] = 0.0
    return dU_dy[1:-1,1:-1,1:-1]
    
  elif axis == 1:  # Evaluate the derivative along x axis
    dU_dx = np.zeros((Ny+2,Nx+2,Nz+2))
    dU_dx[1:-1,1:-1,1:-1] = (U[1:-1,2:,1:-1] - U[1:-1,:-2,1:-1])*0.5*invh
    # Deal with the boundary points, use 1st order biased
    jj_C,ii_C,kk_C = np.where(mask_ex==True)
    maskE = mask_ex[jj_C,ii_C+1,kk_C]
    maskW = mask_ex[jj_C,ii_C-1,kk_C]
    E_boundary = np.where(maskE==False)
    j,i,k = jj_C[E_boundary],ii_C[E_boundary],kk_C[E_boundary]
    dU_dx[j,i,k] = (U[j,i,k] - U[j,i-1,k])*invh
    W_boundary = np.where(maskW==False)
    j,i,k = jj_C[W_boundary],ii_C[W_boundary],kk_C[W_boundary]
    dU_dx[j,i,k] = (U[j,i+1,k] - U[j,i,k])*invh
    EW_boundary = np.where(np.logical_and(maskE==False,maskW==False))
    j,i,k = jj_C[EW_boundary],ii_C[EW_boundary],kk_C[EW_boundary]
    dU_dx[j,i,k] = 0.0
    return dU_dx[1:-1,1:-1,1:-1]
    
  else:  # Evaluate the derivative along z axis
    dU_dz = np.zeros((Ny+2,Nx+2,Nz+2))
    dU_dz[1:-1,1:-1,1:-1] = (U[1:-1,1:-1,2:] - U[1:-1,1:-1,:-2])*0.5*invh
    # Deal with the boundary points, use 1st order biased
    jj_C,ii_C,kk_C = np.where(mask_ex==True)
    maskT = mask_ex[jj_C,ii_C,kk_C+1]
    maskB = mask_ex[jj_C,ii_C,kk_C-1]
    T_boundary = np.where(maskT==False)
    j,i,k = jj_C[T_boundary],ii_C[T_boundary],kk_C[T_boundary]
    dU_dz[j,i,k] = (U[j,i,k] - U[j,i,k-1])*invh
    B_boundary = np.where(maskB==False)
    j,i,k = jj_C[B_boundary],ii_C[B_boundary],kk_C[B_boundary]
    dU_dz[j,i,k] = (U[j,i,k+1] - U[j,i,k])*invh
    TB_boundary = np.where(np.logical_and(maskT==False, maskB==False))
    j,i,k = jj_C[TB_boundary],ii_C[TB_boundary],kk_C[TB_boundary]
    dU_dz[j,i,k] = 0.0
    return dU_dz[1:-1,1:-1,1:-1]
    
    
    
def second_derivative_SOC(U,h,axis=0,mask=None):
  """
  # Evaluate the second derivative of a 3d matrix using second order central difference scheme,
  # biased at the boundaries.
  # The input variable U is the 3D matix containing the data
  # The input h is the grid spacing for the interested axis
  # axis specify the axis for evaluating
  # mask define the mask of the data. If mask is None, the mask is full True.
  Inputs:
    U: 3d field of the variable
    h: grid size. 
    axis: along which dimension the derivative is calculated.
    mask: binary maks of the field.
  Returns:
    dU_dy/dU_dx/dU_dz: 3d field of the derivative.
  """
  
  Ny,Nx,Nz = np.shape(U)
  invh = 1.0/h
  invh2 = invh**2
  
  U_ex = np.zeros((Ny+2,Nx+2,Nz+2))
  U_ex[1:-1,1:-1,1:-1] = U
  U = np.copy(U_ex)
  
  mask_ex = np.zeros((Ny+2,Nx+2,Nz+2)).astype('bool')
  if mask is None:
    mask_ex[1:-1,1:-1,1:-1] = True
  else:
    mask_ex[1:-1,1:-1,1:-1] = mask
    
  if axis == 0:  # The derivative is along y axis.
    d2U_dy2 = np.zeros((Ny+2,Nx+2,Nz+2))
    d2U_dy2[1:-1,1:-1,1:-1] = (U[2:,1:-1,1:-1] + U[:-2,1:-1,1:-1] - 2.0*U[1:-1,1:-1,1:-1])*invh2
    # Deal with the boundary points use biased scheme
    jj_C,ii_C,kk_C = np.where(mask_ex==True)
    maskN = mask_ex[jj_C+1,ii_C,kk_C]
    maskS = mask_ex[jj_C-1,ii_C,kk_C]
    N_boundary = np.where(maskN==False)
    j,i,k = jj_C[N_boundary],ii_C[N_boundary],kk_C[N_boundary]
    d2U_dy2[j,i,k] = (U[j,i,k] + U[j-2,i,k] - 2.0*U[j-1,i,k])*invh2
    S_boundary = np.where(maskS==False)
    j,i,k = jj_C[S_boundary],ii_C[S_boundary],kk_C[S_boundary]
    d2U_dy2[j,i,k] = (U[j,i,k] + U[j+2,i,k] - 2.0*U[j+1,i,k])*invh2
    return d2U_dy2[1:-1,1:-1,1:-1]
    
  elif axis == 1:  # Evaluate the derivative along x axis
    d2U_dx2 = np.zeros((Ny+2,Nx+2,Nz+2))
    d2U_dx2[1:-1,1:-1,1:-1] = (U[1:-1,2:,1:-1] + U[1:-1,:-2,1:-1] - 2.0*U[1:-1,1:-1,1:-1])*invh2
    # Deal with the boundary points use biased scheme
    jj_C,ii_C,kk_C = np.where(mask_ex==True)
    maskE = mask_ex[jj_C,ii_C+1,kk_C]
    maskW = mask_ex[jj_C,ii_C-1,kk_C]
    E_boundary = np.where(maskE==False)
    j,i,k = jj_C[E_boundary],ii_C[E_boundary],kk_C[E_boundary]
    d2U_dx2[j,i,k] = (U[j,i,k] + U[j,i-2,k] - 2.0*U[j,i-1,k])*invh2
    W_boundary = np.where(maskW==False)
    j,i,k = jj_C[W_boundary],ii_C[W_boundary],kk_C[W_boundary]
    d2U_dx2[j,i,k] = (U[j,i,k] + U[j,i+2,k] - 2.0*U[j,i+1,k])*invh2
    return d2U_dx2[1:-1,1:-1,1:-1]
    
  else: # Evaluate the derivative along z axis
    d2U_dz2 = np.zeros((Ny+2,Nx+2,Nz+2))
    d2U_dz2[1:-1,1:-1,1:-1] = (U[1:-1,1:-1,2:] + U[1:-1,1:-1,:-2] - 2.0*U[1:-1,1:-1,1:-1])*invh2
    # Deal with the boundary points use biased scheme
    jj_C,ii_C,kk_C = np.where(mask_ex==True)
    maskT = mask_ex[jj_C,ii_C,kk_C+1]
    maskB = mask_ex[jj_C,ii_C,kk_C-1]
    T_boundary = np.where(maskT==False)
    j,i,k = jj_C[T_boundary],ii_C[T_boundary],kk_C[T_boundary]
    d2U_dz2[j,i,k] = (U[j,i,k] + U[j,i,k-2] - 2.0*U[j,i,k-1])*invh2
    B_boundary = np.where(maskB==False)
    j,i,k = jj_C[B_boundary],ii_C[B_boundary],kk_C[B_boundary]
    d2U_dz2[j,i,k] = (U[j,i,k] + U[j,i,k+2] - 2.0*U[j,i,k+1])*invh2
    return d2U_dz2[1:-1,1:-1,1:-1]
    
    
  
    
    