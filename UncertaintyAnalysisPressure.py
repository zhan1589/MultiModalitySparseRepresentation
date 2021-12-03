import numpy as np
import sys
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg

from NumericalLinearOperators import LinearOperatorGeneration
import NumericalDifference
from EvaluatePressureGradient import NavierStokesMomentum


class PressureUncertaintyEvaluation():
  
  # Evaluate the pressure field and uncertainty of pressure/pressure gradient.
  # The grid setup is non-staggered for pressure
  # The pressure gradient has both staggered and non-staggered setup.
  
  def __init__(self,Xn,Yn,Zn,Un,Vn,Wn,fluid_maskn,nu,rho,dt):
    """
    # Take basic inputs and initilize the flow field.
    # The input of velocity data contains all the snapshots. [Nt,Ny,Nx,Nz]
    # The fluid_maskn is same for all snapshots [Ny,Nx,Nz]
    Inputs:
      Xn,Yn,Zn: 3d mesh grids.
      Un,Vn,Wn: 3d velocity fields given as 4d array with the shape of (Nt,Ny,Nx,Nz)
      fluid_maskn: binary mask of the flow field.
      nu: kinematic viscosity.
      rho: density
      dt: time difference between snapshots.
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
    
    self.nu = nu
    self.rho = rho
    self.mu = self.nu*self.rho
    self.dt = dt
    self.invdt = 1.0/self.dt
    self.invdx = 1.0/self.dx
    self.invdy = 1.0/self.dy
    self.invdz = 1.0/self.dz
    self.invdx2 = self.invdx**2
    self.invdy2 = self.invdy**2
    self.invdz2 = self.invdz**2
    
    # The flow domain is enlarged by adding a layer of ghost cells around the actual domain boundary.
    self.Un = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    self.Vn = np.zeros(self.Un.shape)
    self.Wn = np.zeros(self.Un.shape)
    self.Un[:,1:-1,1:-1,1:-1] = Un
    self.Vn[:,1:-1,1:-1,1:-1] = Vn
    self.Wn[:,1:-1,1:-1,1:-1] = Wn
    self.fluid_mask = np.zeros((self.Ny+2,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask[1:-1,1:-1,1:-1] = fluid_maskn
    
    # Based on the fluid_mask, generates the node index
    self.j,self.i,self.k = np.where(self.fluid_mask==True)
    self.Npts = len(self.j)
    self.fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    self.fluid_index[self.j,self.i,self.k] = range(self.Npts)
    self.iC = self.fluid_index[self.j,self.i,self.k]
    self.iE = self.fluid_index[self.j,self.i+1,self.k]
    self.iW = self.fluid_index[self.j,self.i-1,self.k]
    self.iN = self.fluid_index[self.j+1,self.i,self.k]
    self.iS = self.fluid_index[self.j-1,self.i,self.k]
    self.iT = self.fluid_index[self.j,self.i,self.k+1]
    self.iB = self.fluid_index[self.j,self.i,self.k-1]

    # Generate the fluid mask for dP_dx_staggered and dP_dy_staggered
    self.fluid_mask_x = np.zeros((self.Ny+2,self.Nx+1,self.Nz+2)).astype('bool')
    self.fluid_mask_y = np.zeros((self.Ny+1,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask_z = np.zeros((self.Ny+2,self.Nx+2,self.Nz+1)).astype('bool')
    mask_E = self.fluid_mask[self.j,self.i+1,self.k]
    mask_N = self.fluid_mask[self.j+1,self.i,self.k]
    mask_T = self.fluid_mask[self.j,self.i,self.k+1]
    # For dp_dx locations
    loc = (mask_E==True)
    self.fluid_mask_x[self.j[loc],self.i[loc],self.k[loc]] = True
    # For dp_dy locations
    loc = (mask_N==True)
    self.fluid_mask_y[self.j[loc],self.i[loc],self.k[loc]] = True
    # For dp_dz locations
    loc = (mask_T==True)
    self.fluid_mask_z[self.j[loc],self.i[loc],self.k[loc]] = True
    # Number of points for the staggered arrangements.
    self.j_x,self.i_x,self.k_x = np.where(self.fluid_mask_x==True)
    self.Npts_x = len(self.j_x)
    self.j_y,self.i_y,self.k_y = np.where(self.fluid_mask_y==True)
    self.Npts_y = len(self.j_y)
    self.j_z,self.i_z,self.k_z = np.where(self.fluid_mask_z==True)
    self.Npts_z = len(self.j_z)
    
    # Generate the linear operators that will be used
    LinearOperatorGenerator = LinearOperatorGeneration(Xn,Yn,Zn,fluid_maskn)
    self.O_d_dx = LinearOperatorGenerator.generate_operator_d_dx()
    self.O_d_dy = LinearOperatorGenerator.generate_operator_d_dy()
    self.O_d_dz = LinearOperatorGenerator.generate_operator_d_dz()
    self.O_d_dxdx = LinearOperatorGenerator.generate_operator_d_dx2()
    self.O_d_dydy = LinearOperatorGenerator.generate_operator_d_dy2()
    self.O_d_dzdz = LinearOperatorGenerator.generate_operator_d_dz2()
    self.O_st_x, self.O_st_y, self.O_st_z = LinearOperatorGenerator.generate_operator_collocated_to_staggered()
    self.Identity = LinearOperatorGenerator.generate_operator_temporal()
    self.gradientOperatorLeastSquare = LinearOperatorGenerator.generate_gradient_operator_least_square()
    
    
  def get_staggered_fluid_masks(self):
    # output the fluid mask for staggered arrangements
    return self.fluid_mask_x[1:-1,1:-1,1:-1], self.fluid_mask_y[1:-1,1:-1,1:-1], self.fluid_mask_z[1:-1,1:-1,1:-1]
    
    
  def evaluate_pressure_gradient_staggered(self,Un=None,Vn=None,Wn=None):
    # Evaluate the pressure gradient vectors in the staggered locations using the linear operators.
    if Un is None:
      pass
    else:
      self.Un[:,1:-1,1:-1,1:-1] = Un
      self.Vn[:,1:-1,1:-1,1:-1] = Vn
      self.Wn[:,1:-1,1:-1,1:-1] = Wn

    self.dp_dx_st = np.zeros((self.Nt,self.Ny+2,self.Nx+1,self.Nz+2))
    self.dp_dy_st = np.zeros((self.Nt,self.Ny+1,self.Nx+2,self.Nz+2))
    self.dp_dz_st = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+1))
    # Loop through all snapshots
    for ct in range(self.Nt):
      
      # Write the velocity in the 1D column vector forms.
      u = self.Un[ct,self.j,self.i,self.k]
      v = self.Vn[ct,self.j,self.i,self.k]
      w = self.Wn[ct,self.j,self.i,self.k]
      if ct != 0: # If this is not the first snapshot
        u_minus = self.Un[ct-1,self.j,self.i,self.k]
        v_minus = self.Vn[ct-1,self.j,self.i,self.k]
        w_minus = self.Wn[ct-1,self.j,self.i,self.k]
      if ct != (self.Nt-1):  # If this is not the last snapshot
        u_plus = self.Un[ct+1,self.j,self.i,self.k]
        v_plus = self.Vn[ct+1,self.j,self.i,self.k]
        w_plus = self.Wn[ct+1,self.j,self.i,self.k]

      # Evaluate the temporal derivatives
      if ct == 0:  # If this is the first snapshot
        dp_dx = -self.rho*self.invdt * self.Identity.dot(u_plus)
        dp_dx += self.rho*self.invdt * self.Identity.dot(u)
        dp_dy = -self.rho*self.invdt * self.Identity.dot(v_plus)
        dp_dy += self.rho*self.invdt * self.Identity.dot(v)
        dp_dz = -self.rho*self.invdt * self.Identity.dot(w_plus)
        dp_dz += self.rho*self.invdt * self.Identity.dot(w)
      elif ct == self.Nt-1:  # If this is the last snapshot
        dp_dx = -self.rho*self.invdt * self.Identity.dot(u)
        dp_dx += self.rho*self.invdt * self.Identity.dot(u_minus)
        dp_dy = -self.rho*self.invdt * self.Identity.dot(v)
        dp_dy += self.rho*self.invdt * self.Identity.dot(v_minus)
        dp_dz = -self.rho*self.invdt * self.Identity.dot(w)
        dp_dz += self.rho*self.invdt * self.Identity.dot(w_minus)
      else:  # This is not the first or the last snapshot
        dp_dx = -self.rho*self.invdt*0.5 * self.Identity.dot(u_plus)
        dp_dx += self.rho*self.invdt*0.5 * self.Identity.dot(u_minus)
        dp_dy = -self.rho*self.invdt*0.5 * self.Identity.dot(v_plus)
        dp_dy += self.rho*self.invdt*0.5 * self.Identity.dot(v_minus)
        dp_dz = -self.rho*self.invdt*0.5 * self.Identity.dot(w_plus)
        dp_dz += self.rho*self.invdt*0.5 * self.Identity.dot(w_minus)
        
      # Evaluate the advection terms
      dp_dx += -self.rho*u * (self.O_d_dx.dot(u))
      dp_dx += -self.rho*v * (self.O_d_dy.dot(u))
      dp_dx += -self.rho*w * (self.O_d_dz.dot(u))
      dp_dy += -self.rho*u * (self.O_d_dx.dot(v))
      dp_dy += -self.rho*v * (self.O_d_dy.dot(v))
      dp_dy += -self.rho*w * (self.O_d_dz.dot(v))
      dp_dz += -self.rho*u * (self.O_d_dx.dot(w))
      dp_dz += -self.rho*v * (self.O_d_dy.dot(w))
      dp_dz += -self.rho*w * (self.O_d_dz.dot(w))
      
      # Evalaute the viscous diffusion terms
      dp_dx += self.mu * self.O_d_dxdx.dot(u)
      dp_dx += self.mu * self.O_d_dydy.dot(u)
      dp_dx += self.mu * self.O_d_dzdz.dot(u)
      dp_dy += self.mu * self.O_d_dxdx.dot(v)
      dp_dy += self.mu * self.O_d_dydy.dot(v)
      dp_dy += self.mu * self.O_d_dzdz.dot(v)
      dp_dz += self.mu * self.O_d_dxdx.dot(w)
      dp_dz += self.mu * self.O_d_dydy.dot(w)
      dp_dz += self.mu * self.O_d_dzdz.dot(w)
      
      # Evaluate the values at the staggered locations
      dp_dx_st = self.O_st_x.dot(dp_dx)
      dp_dy_st = self.O_st_y.dot(dp_dy)
      dp_dz_st = self.O_st_z.dot(dp_dz)
      self.dp_dx_st[ct,self.j_x,self.i_x,self.k_x] = dp_dx_st
      self.dp_dy_st[ct,self.j_y,self.i_y,self.k_y] = dp_dy_st
      self.dp_dz_st[ct,self.j_z,self.i_z,self.k_z] = dp_dz_st


    return self.dp_dx_st[:,1:-1,1:-1,1:-1], self.dp_dy_st[:,1:-1,1:-1,1:-1], self.dp_dz_st[:,1:-1,1:-1,1:-1]
    
    
  def evaluate_covariance_pressure_gradient_staggered(self,ct,Un=None,Vn=None,Wn=None,cov=None,cov_minus=None,cov_plus=None):
    # Evaluate the covariance matrix of the staggered pressure gradient field for a single snapshot
    # the input ct indicates which snapshot will be used.
    # The input cov is the covariance matrix for the of velocity (u v w) for the current snapshot.
    # The input cov_minus is the covariance matrix for the velocity (u v w) for last snapshot.
    # The input cov_plus is the covariance matrix for the velocity (u v w) for next snapshot.
    # The input covariace matrix should be in form of sparse matrix (csr).
    # Following the equations and algorithm introduced in the doc file.
    
    # Assign the velocity values to 1d column vectors.
    if Un is None:
      Un_temp = self.Un
      Vn_temp = self.Vn
      Wn_temp = self.Wn
    else:
      Un_temp = np.zeros(self.Un.shape)
      Vn_temp = np.zeros(self.Vn.shape)
      Wn_temp = np.zeros(self.Wn.shape)
      Un_temp[:,1:-1,1:-1,1:-1] = Un
      Vn_temp[:,1:-1,1:-1,1:-1] = Vn 
      Wn_temp[:,1:-1,1:-1,1:-1] = Wn

    u = Un_temp[ct,self.j,self.i,self.k]
    v = Vn_temp[ct,self.j,self.i,self.k]  
    w = Wn_temp[ct,self.j,self.i,self.k] 
    
    # Generate matrix A,B,F,G,L,M
    if ct == 0: # The first snapshot, The B,G,M matrix is zero
      A = -self.rho*self.invdt * self.Identity
      F = -self.rho*self.invdt * self.Identity
      L = -self.rho*self.invdt * self.Identity
      B = scysparse.csr_matrix((self.Npts,self.Npts),dtype=np.float)
      G = scysparse.csr_matrix((self.Npts,self.Npts),dtype=np.float)
      M = scysparse.csr_matrix((self.Npts,self.Npts),dtype=np.float)
    elif ct == (self.Nt-1): # The last snapshot, the matrix A, F, L is zero
      A = scysparse.csr_matrix((self.Npts,self.Npts),dtype=np.float)
      F = scysparse.csr_matrix((self.Npts,self.Npts),dtype=np.float)
      L = scysparse.csr_matrix((self.Npts,self.Npts),dtype=np.float)
      B = self.rho*self.invdt * self.Identity
      G = self.rho*self.invdt * self.Identity
      M = self.rho*self.invdt * self.Identity
    else:  # For the middle snapshots
      A = -0.5*self.rho*self.invdt * self.Identity
      F = -0.5*self.rho*self.invdt * self.Identity
      L = -0.5*self.rho*self.invdt * self.Identity
      B = 0.5*self.rho*self.invdt * self.Identity
      G = 0.5*self.rho*self.invdt * self.Identity
      M = 0.5*self.rho*self.invdt * self.Identity
      
    # Generate matrix C,D,E
    C = -self.rho * scysparse.diags(u,format='csr') * self.O_d_dx
    C += -self.rho * scysparse.diags((self.O_d_dx.dot(u)),format='csr')
    C += -self.rho * scysparse.diags(v,format='csr') * self.O_d_dy
    C += -self.rho * scysparse.diags(w,format='csr') * self.O_d_dz
    C += self.mu * self.O_d_dxdx + self.mu * self.O_d_dydy + self.mu * self.O_d_dzdz
    D = -self.rho * scysparse.diags((self.O_d_dy.dot(u)),format='csr')
    E = -self.rho * scysparse.diags((self.O_d_dz.dot(u)),format='csr')
    # Generate matrix H,J,K
    H = -self.rho * scysparse.diags((self.O_d_dx.dot(v)),format='csr')
    J = -self.rho * scysparse.diags(u,format='csr') * self.O_d_dx
    J += -self.rho * scysparse.diags(v,format='csr') * self.O_d_dy
    J += -self.rho * scysparse.diags((self.O_d_dy.dot(v)),format='csr')
    J += -self.rho * scysparse.diags(w,format='csr') * self.O_d_dz
    J += self.mu * self.O_d_dxdx + self.mu * self.O_d_dydy + self.mu * self.O_d_dzdz
    K = -self.rho * scysparse.diags((self.O_d_dz.dot(v)),format='csr')
    # Generate matrix N,P,Q
    N = -self.rho * scysparse.diags((self.O_d_dx.dot(w)),format='csr')
    P = -self.rho * scysparse.diags((self.O_d_dy.dot(w)),format='csr')
    Q = -self.rho * scysparse.diags(u,format='csr') * self.O_d_dx
    Q += -self.rho * scysparse.diags(v,format='csr') * self.O_d_dy
    Q += -self.rho * scysparse.diags((self.O_d_dz.dot(w)),format='csr')
    Q += -self.rho * scysparse.diags(w,format='csr') * self.O_d_dz
    Q += self.mu * self.O_d_dxdx + self.mu * self.O_d_dydy + self.mu * self.O_d_dzdz

    if ct == 0: # The first snapshot
      C += self.rho*self.invdt * self.Identity  # For the temporal derivative
      J += self.rho*self.invdt * self.Identity
      Q += self.rho*self.invdt * self.Identity
    elif ct == (self.Nt-1): # The last snapshot
      C += -self.rho*self.invdt * self.Identity
      J += -self.rho*self.invdt * self.Identity 
      Q += -self.rho*self.invdt * self.Identity
      
    # Generate the larger staggering matrix
    O_st = scysparse.csr_matrix((self.Npts_x+self.Npts_y+self.Npts_z,self.Npts*3),dtype=np.float)
    O_st_x_coo = self.O_st_x.tocoo()
    O_st_x_data = O_st_x_coo.data
    O_st_x_row = O_st_x_coo.row
    O_st_x_col = O_st_x_coo.col
    O_st_y_coo = self.O_st_y.tocoo()
    O_st_y_data = O_st_y_coo.data
    O_st_y_row = O_st_y_coo.row + self.Npts_x
    O_st_y_col = O_st_y_coo.col + self.Npts
    O_st_z_coo = self.O_st_z.tocoo()
    O_st_z_data = O_st_z_coo.data
    O_st_z_row = O_st_z_coo.row + self.Npts_x + self.Npts_y
    O_st_z_col = O_st_z_coo.col + self.Npts*2
    O_st[O_st_x_row,O_st_x_col] = O_st_x_data
    O_st[O_st_y_row,O_st_y_col] = O_st_y_data
    O_st[O_st_z_row,O_st_z_col] = O_st_z_data

    # Generate the matrix R,S,T
    # For matrix R
    AFL = scysparse.csr_matrix((self.Npts*3,self.Npts*3),dtype=np.float)
    A_coo = A.tocoo()
    A_data = A_coo.data
    A_row = A_coo.row
    A_col = A_coo.col
    F_coo = F.tocoo()
    F_data = F_coo.data
    F_row = F_coo.row + self.Npts
    F_col = F_coo.col + self.Npts
    L_coo = L.tocoo()
    L_data = L_coo.data
    L_row = L_coo.row + self.Npts*2
    L_col = L_coo.col + self.Npts*2
    AFL[A_row,A_col] = A_data
    AFL[F_row,F_col] = F_data
    AFL[L_row,L_col] = L_data
    R = O_st*AFL
    # For matrix S
    BGM = scysparse.csr_matrix((self.Npts*3,self.Npts*3),dtype=np.float)
    B_coo = B.tocoo()
    B_data = B_coo.data
    B_row = B_coo.row
    B_col = B_coo.col
    G_coo = G.tocoo()
    G_data = G_coo.data
    G_row = G_coo.row + self.Npts
    G_col = G_coo.col + self.Npts
    M_coo = M.tocoo()
    M_data = M_coo.data
    M_row = M_coo.row + self.Npts*2
    M_col = M_coo.col + self.Npts*2
    BGM[B_row,B_col] = B_data
    BGM[G_row,G_col] = G_data
    BGM[M_row,M_col] = M_data
    S = O_st*BGM
    # For matrix T
    CJQ = scysparse.csr_matrix((self.Npts*3,self.Npts*3),dtype=np.float)
    #CJQ[:self.Npts,:self.Npts] = C
    C_coo = C.tocoo()
    C_data = C_coo.data
    C_row = C_coo.row
    C_col = C_coo.col
    CJQ[C_row,C_col] = C_data
    #CJQ[:self.Npts,self.Npts:self.Npts*2] = D
    D_coo = D.tocoo()
    D_data = D_coo.data
    D_row = D_coo.row
    D_col = D_coo.col + self.Npts
    CJQ[D_row,D_col] = D_data
    #CJQ[:self.Npts,self.Npts*2:] = E
    E_coo = E.tocoo()
    E_data = E_coo.data
    E_row = E_coo.row
    E_col = E_coo.col + self.Npts*2
    CJQ[E_row,E_col] = E_data
    #CJQ[self.Npts:self.Npts*2,:self.Npts] = H
    H_coo = H.tocoo()
    H_data = H_coo.data
    H_row = H_coo.row + self.Npts
    H_col = H_coo.col
    CJQ[H_row,H_col] = H_data
    #CJQ[self.Npts:self.Npts*2,self.Npts:self.Npts*2] = J
    J_coo = J.tocoo()
    J_data = J_coo.data
    J_row = J_coo.row + self.Npts
    J_col = J_coo.col + self.Npts
    CJQ[J_row,J_col] = J_data
    #CJQ[self.Npts:self.Npts*2,self.Npts*2:] = K
    K_coo = K.tocoo()
    K_data = K_coo.data
    K_row = K_coo.row + self.Npts
    K_col = K_coo.col + self.Npts*2
    CJQ[K_row,K_col] = K_data
    #CJQ[self.Npts*2:,:self.Npts] = N
    N_coo = N.tocoo()
    N_data = N_coo.data
    N_row = N_coo.row + self.Npts*2
    N_col = N_coo.col 
    CJQ[N_row,N_col] = N_data
    #CJQ[self.Npts*2:,self.Npts:self.Npts*2] = P
    P_coo = P.tocoo()
    P_data = P_coo.data
    P_row = P_coo.row + self.Npts*2
    P_col = P_coo.col + self.Npts
    CJQ[P_row,P_col] = P_data
    #CJQ[self.Npts*2:,self.Npts*2:] = Q
    Q_coo = Q.tocoo()
    Q_data = Q_coo.data
    Q_row = Q_coo.row + self.Npts*2
    Q_col = Q_coo.col + self.Npts*2
    CJQ[Q_row,Q_col] = Q_data
    # For T matrix
    T = O_st*CJQ
    
    # Evaluate the covariace of the staggered pressure gradients based on the covariance matrices.
    if ct == 0: # The first snapshot
      cov_minus = scysparse.csr_matrix((self.Npts*3,self.Npts*3),dtype=np.float)
    elif ct == (self.Nt-1): # The last snapshot
      cov_plus = scysparse.csr_matrix((self.Npts*3,self.Npts*3),dtype=np.float)
    
    self.cov_pgrad_st = R*cov_plus*R.transpose() + S*cov_minus*S.transpose() + T*cov*T.transpose()
    
    return self.cov_pgrad_st
    
    
  def evaluate_pressure_generalized_least_square(self,pgrad_st,cov_pgrad_st=None,sparse_treatment='augmented',ref_point=None,sparse_solver='spsolve'):
    # Evaluate the pressure results with pressure gradient values and its covariance matrix.
    # Using the generalized least sqaure method.
    # The input pgrad_st is the 1d vector contains the values of dp/dx dp/dy and dp/dz at staggered locations.
    # The input cov_pgrad_st is the covaraince matrix (sparse) of the pgrad_st values.
    # The sparse_treatment is the way of dealing the inverse of covariance matrix (full matrix in theory)
    
    # If the input covarinae matrix is None, use the ordinary least square
    if cov_pgrad_st is None:
      cov_pgrad_st = scysparse.eye(self.Npts_x+self.Npts_y+self.Npts_z,dtype=np.float,format='csr')
    
    # Dealing with the inverse of the covarinace matrix.
    if sparse_treatment == 'direct':  # Compute the inverse directly.
      inv_cov = splinalg.inv(cov_pgrad_st)
      # Generate the LHS operator
      LHS = self.gradientOperatorLeastSquare.transpose() * inv_cov * self.gradientOperatorLeastSquare

      # Generate the RHS 
      RHS = self.gradientOperatorLeastSquare.transpose() * (splinalg.spsolve(cov_pgrad_st,pgrad_st))
    
      # Assign the reference point 
      if ref_point is None:
        ref_point = [self.j[0]-1,self.i[0]-1,self.k[0]-1]

      j_ref = ref_point[0]+1
      i_ref = ref_point[1]+1
      k_ref = ref_point[2]+1
      ref_index = self.fluid_index[j_ref,i_ref,k_ref]
      LHS[ref_index,:] = 0.0
      LHS[ref_index,ref_index] = 1.0
      LHS.eliminate_zeros()
      RHS[ref_index] = 0.0
      
      # Solve for pressure results
      p_vector = splinalg.spsolve(LHS,RHS)
      

    elif sparse_treatment == 'augmented':
      # Construct the sparst augmented linear system and solve for pressure
      gradientOperatorLeastSquare_coo = self.gradientOperatorLeastSquare.tocoo()
      G_data = gradientOperatorLeastSquare_coo.data
      G_row = gradientOperatorLeastSquare_coo.row
      G_col = gradientOperatorLeastSquare_coo.col
      gradientOperatorLeastSquare_large = scysparse.csr_matrix((self.Npts_x+self.Npts_y+self.Npts_z+1,self.Npts),dtype=np.float)
      gradientOperatorLeastSquare_large[G_row,G_col] = G_data
      gradientOperatorLeastSquare_large[-1,0] = 1.0
      # Modify the covariace matrix based on the augmented linear system
      cov_pgrad_st_coo = cov_pgrad_st.tocoo()
      W_data = cov_pgrad_st_coo.data
      W_row = cov_pgrad_st_coo.row
      W_col = cov_pgrad_st_coo.col
      cov_pgrad_st_large = scysparse.csr_matrix((self.Npts_x+self.Npts_y+self.Npts_z+1,self.Npts_x+self.Npts_y+self.Npts_z+1),dtype=np.float)
      cov_pgrad_st_large[W_row,W_col] = W_data
      cov_pgrad_st_large[-1,-1] = 1.0
      # Modify the rhs vector based on the augmented linear system
      pgrad_st_large = np.zeros(self.Npts_x+self.Npts_y+self.Npts_z+1)
      pgrad_st_large[:-1] = pgrad_st
      # Construct the augmented linear system
      fundamental_matrix = scysparse.csr_matrix((self.Npts_x+self.Npts_y+self.Npts_z+1+self.Npts,self.Npts_x+self.Npts_y+self.Npts_z+1+self.Npts),dtype=np.float)
      W_coo = cov_pgrad_st_large.tocoo()
      N_W_rows, N_W_cols = W_coo.shape
      W_data = W_coo.data
      W_row = W_coo.row
      W_col = W_coo.col
      fundamental_matrix[W_row,W_col] = W_data
      A_coo = gradientOperatorLeastSquare_large.tocoo()
      N_A_rows, N_A_cols = A_coo.shape
      A_data = A_coo.data
      A_row = A_coo.row
      A_col = A_coo.col + N_W_cols
      fundamental_matrix[A_row,A_col] = A_data
      AT_coo = (gradientOperatorLeastSquare_large.transpose()).tocoo()
      N_AT_rows, N_AT_cols = AT_coo.shape
      AT_data = AT_coo.data
      AT_row = AT_coo.row + N_W_rows
      AT_col = AT_coo.col
      fundamental_matrix[AT_row,AT_col] = AT_data
      # RHS of the augmented linear system
      fundamental_rhs = np.zeros(N_W_rows + N_AT_rows)
      fundamental_rhs[:N_W_rows] = pgrad_st_large
      # Prepare an initla guess of the solution of the linear system for iteration solvers.
      # The initial guess is the solution by ordinary least squares.
      yx_vector_0 = None
      '''
      if sparse_solver != 'spsolve':
        OLS_LHS = gradientOperatorLeastSquare_large.transpose() * gradientOperatorLeastSquare_large
        OLS_RHS = (gradientOperatorLeastSquare_large.transpose()).dot(pgrad_st_large)
        p_vector_ols = splinalg.spsolve(OLS_LHS,OLS_RHS)
        yx_vector_0 = np.zeros(N_W_rows + N_AT_rows)
        yx_vector_0[N_W_rows:] = p_vector_ols
      '''
      # Solve the augmented system
      if sparse_solver == 'spsolve':
        yx_vector = splinalg.spsolve(fundamental_matrix,fundamental_rhs)
      elif sparse_solver == 'bicg':
        yx_vector,info = splinalg.bicgstab(fundamental_matrix,fundamental_rhs,x0=yx_vector_0)
        print('  Covergence info = '+str(info))
      elif sparse_solver == 'cg':
        yx_vector,info = splinalg.cg(fundamental_matrix,fundamental_rhs,x0=yx_vector_0)
        print('  Covergence info = '+str(info))
      elif sparse_solver == 'lgmres':
        yx_vector,info = splinalg.lgmres(fundamental_matrix,fundamental_rhs,x0=yx_vector_0)
        print('  Covergence info = '+str(info))


      p_vector = yx_vector[N_W_rows:]

    
    # Assign to the 3d field.
    pressure = np.zeros(self.fluid_mask.shape)
    pressure[self.j,self.i,self.k] = p_vector
    
    return pressure[1:-1,1:-1,1:-1]

  
  def evaluate_pressure_ordinary_least_square(self,pgrad_st_stack,ref_point=None):
    # Solve the pressure field using ordinary least squares

    pressure = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    
    LHS = self.gradientOperatorLeastSquare.transpose() * self.gradientOperatorLeastSquare

    if ref_point is None:
      ref_point = [self.j[0]-1,self.i[0]-1,self.k[0]-1]
    j_ref = ref_point[0]+1
    i_ref = ref_point[1]+1
    k_ref = ref_point[2]+1
    ref_index = self.fluid_index[j_ref,i_ref,k_ref]
    LHS[ref_index,:] = 0.0
    LHS[ref_index,ref_index] = 1.0
    LHS.eliminate_zeros()
    # LU decomposition
    LHS_LU = splinalg.splu(LHS)
    
    # Generate the RHS 
    for ct in range(self.Nt):
      pgrad_st = pgrad_st_stack[ct]
      RHS = self.gradientOperatorLeastSquare.transpose().dot(pgrad_st)
      RHS[ref_index] = 0.0

      # Solve for pressure results
      p_vector = LHS_LU.solve(RHS)

      # Assign to the 3d field.
      pressure[ct,self.j,self.i,self.k] = p_vector
    
    return pressure[:,1:-1,1:-1,1:-1]



  def evaluate_pressure_weighted_least_square(self,pgrad_st,cov_pgrad_st,ref_point=None):
    # Solve the pressure field using weighted least-squares

    inv_cov = cov_pgrad_st.power(-1)
    # Generate the LHS operator
    LHS = self.gradientOperatorLeastSquare.transpose() * inv_cov * self.gradientOperatorLeastSquare

    # Generate the RHS 
    RHS = (self.gradientOperatorLeastSquare.transpose() * inv_cov).dot(pgrad_st)
  
    # Assign the reference point 
    if ref_point is None:
      ref_point = [self.j[0]-1,self.i[0]-1,self.k[0]-1]

    j_ref = ref_point[0]+1
    i_ref = ref_point[1]+1
    k_ref = ref_point[2]+1
    ref_index = self.fluid_index[j_ref,i_ref,k_ref]
    LHS[ref_index,:] = 0.0
    LHS[ref_index,ref_index] = 1.0
    LHS.eliminate_zeros()
    RHS[ref_index] = 0.0
    
    # Solve for pressure results
    p_vector = splinalg.spsolve(LHS,RHS)

    # Assign to the 3d field.
    pressure = np.zeros(self.fluid_mask.shape)
    pressure[self.j,self.i,self.k] = p_vector
    
    return pressure[1:-1,1:-1,1:-1]

    
  def evaluate_pressure_Poisson(self):
    # Solve for the pressure field using a Poison solver
    # The result will be used for the CG method
    
    # Evaluate the pressure gradient
    Un = self.Un[:,1:-1,1:-1,1:-1]
    Vn = self.Vn[:,1:-1,1:-1,1:-1]
    Wn = self.Wn[:,1:-1,1:-1,1:-1]
    pressure_gradient_calculator = NavierStokesMomentum(self.Xn,self.Yn,self.Zn,Un,Vn,Wn,self.fluid_mask[1:-1,1:-1,1:-1],self.nu,self.rho,self.dt)
    dP_dx, dP_dy, dP_dz = pressure_gradient_calculator.eval_pressure_gradient()
    
    # Evaluate the source term of the Poisson equation (divergence of pressure gradient)
    div_pgrad = np.zeros((self.Nt,self.Ny,self.Nx,self.Nz))
    for ct in range(self.Nt):
      div_pgrad[ct] = NumericalDifference.first_derivative_SOC(dP_dx[ct],self.dx,axis=1,mask=self.fluid_mask[1:-1,1:-1,1:-1])
      div_pgrad[ct] += NumericalDifference.first_derivative_SOC(dP_dy[ct],self.dy,axis=0,mask=self.fluid_mask[1:-1,1:-1,1:-1])
      div_pgrad[ct] += NumericalDifference.first_derivative_SOC(dP_dz[ct],self.dz,axis=2,mask=self.fluid_mask[1:-1,1:-1,1:-1])
    
    # Generate the linear operatrors.
    LinearOperatorGenerator = LinearOperatorGeneration(self.Xn,self.Yn,self.Zn,self.fluid_mask[1:-1,1:-1,1:-1])
    LaplacianOperator, RHS = LinearOperatorGenerator.generate_laplacian_operator_rhs_neumann(div_pgrad,dP_dx,dP_dy,dP_dz)
    
    # Perform LU decomposition
    print(' Performing LU decomposition')
    OperatorLU = splinalg.splu(LaplacianOperator)
    
    # Solve for the field
    Pn_Poisson = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    
    for ct in range(self.Nt):
      print('  Solve for ct = '+str(ct))
      Pn_vector = OperatorLU.solve(RHS[ct])
      Pn_Poisson[ct,self.j,self.i,self.k] = Pn_vector

    return Pn_Poisson[:,1:-1,1:-1,1:-1]








    