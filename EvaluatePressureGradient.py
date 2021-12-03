import numpy as np
import sys

sys.path.insert(0, '../')
import NumericalDifference

class NavierStokesMomentum():
  # This module calculates pressure gradient field from given velocity fields.

  def __init__(self,Xn,Yn,Zn,Un,Vn,Wn,fluid_maskn,nu,rho,dt,gravity_direction=None):
    """
    # take inputs
    Inputs:
      Xn,Yn,Zn: 3d mesh grids
      fluid_maskn: binary mask of flow field.
      Un,vn,Wn: 3d velocity fields given as 4d array with the shape (Nt,Ny,Nx,Nz)
      nu: kinematic viscosity
      rho: density
      dt: time difference between snapshots
      gravity_direction: direction of the gravity acceleration. None means not considered.
    """
    
    self.Xn = Xn  
    self.Yn = Yn
    self.Zn = Zn
    self.xn = self.Xn[0,:,0]
    self.yn = self.Yn[:,0,0]
    self.zn = self.Zn[0,0,:]
    self.dx = self.xn[1] - self.xn[0]
    self.dy = self.yn[1] - self.yn[0]
    self.dz = self.zn[1] - self.zn[0]
    self.Nt = np.shape(Un)[0]
    self.Nx = len(self.xn)
    self.Ny = len(self.yn)
    self.Nz = len(self.zn)
    self.invdx = 1.0/self.dx
    self.invdy = 1.0/self.dy
    self.invdz = 1.0/self.dz
    self.invdx2 = self.invdx**2
    self.invdy2 = self.invdy**2
    self.invdz2 = self.invdz**2
    self.dt = dt
    self.invdt = 1.0/self.dt
    
    self.nu = nu
    self.rho = rho
    self.dt = dt
    
    self.U = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    self.V = np.zeros(self.U.shape)
    self.W = np.zeros(self.U.shape)
    self.U[:,1:-1,1:-1,1:-1] = Un
    self.V[:,1:-1,1:-1,1:-1] = Vn
    self.W[:,1:-1,1:-1,1:-1] = Wn
    
    g_constant = 9.8
    if gravity_direction is None:
      self.g_x = 0.0
      self.g_y = 0.0
      self.g_z = 0.0
    else:
      g_i = gravity_direction[0]
      g_j = gravity_direction[1]
      g_k = gravity_direction[2]
      self.g_x = g_constant * g_i / (g_i**2 + g_j**2 + g_k**2)**0.5
      self.g_y = g_constant * g_j / (g_i**2 + g_j**2 + g_k**2)**0.5
      self.g_z = g_constant * g_k / (g_i**2 + g_j**2 + g_k**2)**0.5
    
    self.fluid_mask = np.zeros((self.Ny+2,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask[1:-1,1:-1,1:-1] = fluid_maskn

  
  def eval_temporal_derivative(self):
    # Evaluate the temporal derivatives with 2nd order central difference.
    
    self.dU_dt = np.zeros(self.U.shape)
    self.dV_dt = np.zeros(self.V.shape)
    self.dW_dt = np.zeros(self.W.shape)
    # For the middle time steps, 2nd order central difference is employed
    self.dU_dt[1:-1,:,:,:] = (self.U[2:,:,:,:] - self.U[:-2,:,:,:])*0.5*self.invdt
    self.dV_dt[1:-1,:,:,:] = (self.V[2:,:,:,:] - self.V[:-2,:,:,:])*0.5*self.invdt
    self.dW_dt[1:-1,:,:,:] = (self.W[2:,:,:,:] - self.W[:-2,:,:,:])*0.5*self.invdt
    # For the first and last time steps, 1st order biased is employed
    self.dU_dt[0,:,:,:] = (self.U[1,:,:,:] - self.U[0,:,:,:])*self.invdt
    self.dV_dt[0,:,:,:] = (self.V[1,:,:,:] - self.V[0,:,:,:])*self.invdt
    self.dW_dt[0,:,:,:] = (self.W[1,:,:,:] - self.W[0,:,:,:])*self.invdt
    self.dU_dt[-1,:,:,:] = (self.U[-1,:,:,:] - self.U[-2,:,:,:])*self.invdt
    self.dV_dt[-1,:,:,:] = (self.V[-1,:,:,:] - self.V[-2,:,:,:])*self.invdt
    self.dW_dt[-1,:,:,:] = (self.W[-1,:,:,:] - self.W[-2,:,:,:])*self.invdt


  def eval_advection_term(self):
    # Evaluate the advection term in NS equation (non-conservative form)
    # The advection terms UdU/dx and VdU/dy and WdU/dz related to the dP/dx
    # The advection terms UdV/dx and VdV/dy and WdV/dz related to the dP/dy
    # The advection terms UdW/dx and VdW/dy and WdW/dz related to the dP/dz
    
    self.R_i = np.zeros(self.U.shape)  # x component of advection term
    self.R_j = np.zeros(self.V.shape)  # y component of advection term
    self.R_k = np.zeros(self.W.shape)  # z component of advection term
    
    for ct in range(self.Nt):
      
      dU_dx = NumericalDifference.first_derivative_SOC(self.U[ct],self.dx,axis=1,mask=self.fluid_mask)
      UdU_dx = self.U[ct]*dU_dx
      
      dU_dy = NumericalDifference.first_derivative_SOC(self.U[ct],self.dy,axis=0,mask=self.fluid_mask)
      VdU_dy = self.V[ct]*dU_dy

      dU_dz = NumericalDifference.first_derivative_SOC(self.U[ct],self.dz,axis=2,mask=self.fluid_mask)
      WdU_dz = self.W[ct]*dU_dz
      
      dV_dx = NumericalDifference.first_derivative_SOC(self.V[ct],self.dx,axis=1,mask=self.fluid_mask)
      UdV_dx = self.U[ct]*dV_dx
      
      dV_dy = NumericalDifference.first_derivative_SOC(self.V[ct],self.dy,axis=0,mask=self.fluid_mask)
      VdV_dy = self.V[ct]*dV_dy

      dV_dz = NumericalDifference.first_derivative_SOC(self.V[ct],self.dz,axis=2,mask=self.fluid_mask)
      WdV_dz = self.W[ct]*dV_dz

      dW_dx = NumericalDifference.first_derivative_SOC(self.W[ct],self.dx,axis=1,mask=self.fluid_mask)
      UdW_dx = self.U[ct]*dW_dx
      
      dW_dy = NumericalDifference.first_derivative_SOC(self.W[ct],self.dy,axis=0,mask=self.fluid_mask)
      VdW_dy = self.V[ct]*dW_dy

      dW_dz = NumericalDifference.first_derivative_SOC(self.W[ct],self.dz,axis=2,mask=self.fluid_mask)
      WdW_dz = self.W[ct]*dW_dz
    
      self.R_i[ct] = self.rho*(UdU_dx + VdU_dy + WdU_dz)
      self.R_j[ct] = self.rho*(UdV_dx + VdV_dy + WdV_dz)
      self.R_k[ct] = self.rho*(UdW_dx + VdW_dy + WdW_dz)


  def eval_viscous_term(self):
    # Evaluate the viscous diffusion terms
    # dU/dxdx, dU/dydy, dU/dzdz are related to dP/dx 
    # dV/dxdx, dV/dydy, dV/dzdz are related to dP/dy 
    # dW/dxdx, dW/dydy, dW/dzdz are related to dP/dz
    
    self.Vis_i = np.zeros(self.U.shape)  # x component of the vsicous term
    self.Vis_j = np.zeros(self.V.shape)  # y component of the viscous term
    self.Vis_k = np.zeros(self.W.shape)  # z component of the viscous term
    
    for ct in range(self.Nt):
      self.dU_dxdx = NumericalDifference.second_derivative_SOC(self.U[ct],self.dx,axis=1,mask=self.fluid_mask)
      self.dU_dydy = NumericalDifference.second_derivative_SOC(self.U[ct],self.dy,axis=0,mask=self.fluid_mask)
      self.dU_dzdz = NumericalDifference.second_derivative_SOC(self.U[ct],self.dz,axis=2,mask=self.fluid_mask)
      self.dV_dxdx = NumericalDifference.second_derivative_SOC(self.V[ct],self.dx,axis=1,mask=self.fluid_mask)
      self.dV_dydy = NumericalDifference.second_derivative_SOC(self.V[ct],self.dy,axis=0,mask=self.fluid_mask)
      self.dV_dzdz = NumericalDifference.second_derivative_SOC(self.V[ct],self.dz,axis=2,mask=self.fluid_mask)
      self.dW_dxdx = NumericalDifference.second_derivative_SOC(self.W[ct],self.dx,axis=1,mask=self.fluid_mask)
      self.dW_dydy = NumericalDifference.second_derivative_SOC(self.W[ct],self.dy,axis=0,mask=self.fluid_mask)
      self.dW_dzdz = NumericalDifference.second_derivative_SOC(self.W[ct],self.dz,axis=2,mask=self.fluid_mask)
      
      self.Vis_i[ct] = self.nu*self.rho*(self.dU_dxdx + self.dU_dydy + self.dU_dzdz)
      self.Vis_j[ct] = self.nu*self.rho*(self.dV_dxdx + self.dV_dydy + self.dV_dzdz)
      self.Vis_k[ct] = self.nu*self.rho*(self.dW_dxdx + self.dW_dydy + self.dW_dzdz)


  def eval_pressure_gradient(self):
    # Evaluate the pressure gradient
    
    self.eval_temporal_derivative()
    self.eval_advection_term()
    self.eval_viscous_term()
    
    self.dP_dx = self.Vis_i - self.rho*self.dU_dt - self.R_i + self.rho*self.g_x
    self.dP_dy = self.Vis_j - self.rho*self.dV_dt - self.R_j + self.rho*self.g_y
    self.dP_dz = self.Vis_k - self.rho*self.dW_dt - self.R_k + self.rho*self.g_z
    
    return self.dP_dx[:,1:-1,1:-1,1:-1], self.dP_dy[:,1:-1,1:-1,1:-1], self.dP_dz[:,1:-1,1:-1,1:-1]


  def eval_pressure_gradient_staggered(self):
    # Evaluate the pressure gradient at the staggered locations
    # Also generate the fluid mask for the staggered pressure gradients.
    
    self.dP_dx = np.zeros(self.U.shape)
    self.dP_dy = np.zeros(self.V.shape)
    self.dP_dz = np.zeros(self.W.shape)
    self.dP_dx[:,1:-1,1:-1,1:-1], self.dP_dy[:,1:-1,1:-1,1:-1], self.dP_dz[:,1:-1,1:-1,1:-1] = self.eval_pressure_gradient()
    
    self.dP_dx_staggered = 0.5*(self.dP_dx[:,:,1:,:] + self.dP_dx[:,:,:-1,:])
    self.dP_dy_staggered = 0.5*(self.dP_dy[:,1:,:,:] + self.dP_dy[:,:-1,:,:])
    self.dP_dz_staggered = 0.5*(self.dP_dz[:,:,:,1:] + self.dP_dz[:,:,:,:-1])
    # Generate the fluid mask for dP_dx_staggered and dP_dy_staggered
    self.fluid_mask_dp_dx = np.zeros((self.Ny+2,self.Nx+1,self.Nz+2)).astype('bool')
    self.fluid_mask_dp_dy = np.zeros((self.Ny+1,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask_dp_dz = np.zeros((self.Ny+2,self.Nx+2,self.Nz+1)).astype('bool')
    j,i,k = np.where(self.fluid_mask==True)
    mask_E = self.fluid_mask[j,i+1,k]
    mask_N = self.fluid_mask[j+1,i,k]
    mask_T = self.fluid_mask[j,i,k+1]
    # For dp_dx locations
    loc = (mask_E==True)
    self.fluid_mask_dp_dx[j[loc],i[loc],k[loc]] = True
    # For dp_dy locations
    loc = (mask_N==True)
    self.fluid_mask_dp_dy[j[loc],i[loc],k[loc]] = True
    # For dp_dz locations
    loc = (mask_T==True)
    self.fluid_mask_dp_dz[j[loc],i[loc],k[loc]] = True
    
    return self.dP_dx_staggered[:,1:-1,1:-1,1:-1], self.dP_dy_staggered[:,1:-1,1:-1,1:-1], self.dP_dz_staggered[:,1:-1,1:-1,1:-1], \
           self.fluid_mask_dp_dx[1:-1,1:-1,1:-1], self.fluid_mask_dp_dy[1:-1,1:-1,1:-1], self.fluid_mask_dp_dz[1:-1,1:-1,1:-1]















