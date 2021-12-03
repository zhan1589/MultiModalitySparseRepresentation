import numpy as np
import sys
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg

'''
generate linear operators for pressure and flow reconstruction.
include the Laplacian operator for Poisson solver (Neumann BC and Dirichlet BC),
the gradient operator and Least square operator mainly for WLS pressure reconstruction
Also the operator which converts the pressure gradients to pressure RHS.

'''

class LinearOperatorGeneration():
  
  def __init__(self,Xn,Yn,Zn,fluid_maskn,BC='Dirichlet'):
    """
    # take inputs
    Inputs:
        Xn,Yn,Zn: 3d mesh grids.
        fluid_maskn: the binary flow mask.
        BC: boundary condition, either Dirichlet or Neumann
    """
    
    self.Xn = Xn
    self.Yn = Yn
    self.Zn = Zn
    self.xn = self.Xn[0,:,0]
    self.yn = self.Yn[:,0,0]
    self.zn = self.Zn[0,0,:]
    self.Nx = len(self.xn)
    self.Ny = len(self.yn)
    self.Nz = len(self.zn)
    self.dx = self.xn[1] - self.xn[0]
    self.dy = self.yn[1] - self.yn[0]
    self.dz = self.zn[1] - self.zn[0]
    self.invdx = 1.0/self.dx
    self.invdy = 1.0/self.dy
    self.invdz = 1.0/self.dz
    self.invdx2 = self.invdx**2
    self.invdy2 = self.invdy**2
    self.invdz2 = self.invdz**2
    
    self.fluid_mask = np.zeros((self.Ny+2,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask[1:-1,1:-1,1:-1] = fluid_maskn
    
    self.BC = BC
    
    
  def generate_operator_d_dx(self):
    # Generate a linear operator which dot with vector returns value of d/dx
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iE = fluid_index[j,i+1,k]
    iW = fluid_index[j,i-1,k]
    self.OperatorDDX = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    
    # if the east and west nodes are inside domain, using the second order central differencing
    loc = (iE!=-1)*(iW!=-1)
    self.OperatorDDX[iC[loc],iE[loc]] += 0.5*self.invdx
    self.OperatorDDX[iC[loc],iW[loc]] += -0.5*self.invdx
    # For the boundary ponits, use 1st oder biased.
    # if the east node is outside domain
    loc = (iE==-1)
    self.OperatorDDX[iC[loc],iC[loc]] += 1.0*self.invdx
    self.OperatorDDX[iC[loc],iW[loc]] += -1.0*self.invdx
    # if the west node is outside domain
    loc = (iW==-1)
    self.OperatorDDX[iC[loc],iE[loc]] += 1.0*self.invdx
    self.OperatorDDX[iC[loc],iC[loc]] += -1.0*self.invdx
    
    return self.OperatorDDX  

  
  def generate_operator_d_dy(self):
    # Generate a linear operator which dot with vector returns value of d/dy    
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iN = fluid_index[j+1,i,k]
    iS = fluid_index[j-1,i,k]
    self.OperatorDDY = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    
    # If the north and south nodes are inside domain, using the second order central differencing
    loc = (iN!=-1)*(iS!=-1)
    self.OperatorDDY[iC[loc],iN[loc]] += 0.5*self.invdy
    self.OperatorDDY[iC[loc],iS[loc]] += -0.5*self.invdy
    # For the boundary points, use 1st order biased
    # if the north node is outside domain
    loc = (iN==-1)
    self.OperatorDDY[iC[loc],iC[loc]] += 1.0*self.invdy
    self.OperatorDDY[iC[loc],iS[loc]] += -1.0*self.invdy
    # if the south node is outside domain
    loc = (iS==-1)
    self.OperatorDDY[iC[loc],iN[loc]] += 1.0*self.invdy
    self.OperatorDDY[iC[loc],iC[loc]] += -1.0*self.invdy

    return self.OperatorDDY
    
    
  def generate_operator_d_dz(self):
    # Generate a linear operator which dot with vector returns value of d/dy    
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iT = fluid_index[j,i,k+1]
    iB = fluid_index[j,i,k-1]
    self.OperatorDDZ = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    
    # If the top and bottom nodes are inside domain, using the second order central differencing
    loc = (iT!=-1)*(iB!=-1)
    self.OperatorDDZ[iC[loc],iT[loc]] += 0.5*self.invdz
    self.OperatorDDZ[iC[loc],iB[loc]] += -0.5*self.invdz
    # For the boundary points, use 1st order biased
    # if the top node is outside domain
    loc = (iT==-1)
    self.OperatorDDZ[iC[loc],iC[loc]] += 1.0*self.invdz
    self.OperatorDDZ[iC[loc],iB[loc]] += -1.0*self.invdz
    # if the bottom node is outside domain
    loc = (iB==-1)
    self.OperatorDDZ[iC[loc],iT[loc]] += 1.0*self.invdz
    self.OperatorDDZ[iC[loc],iC[loc]] += -1.0*self.invdz

    return self.OperatorDDZ

  def generate_operator_gradient_two_masks(self,mask_in):
    # Generate the linear operator which dot with vector on "mask_in" gives gradients on "fluid_mask"
    # The shape of mask_in should be same as fluid_maskn input at the beginning.
    self.mask_in = np.zeros(self.fluid_mask.shape).astype('bool')
    self.mask_in[1:-1,1:-1,1:-1] = mask_in

    # For the indexing of fluid_mask
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    # For the indexing of mask_in
    j_in, i_in, k_in = np.where(self.mask_in==True)
    Npts_in = len(j_in)
    fluid_index_in = -np.ones(self.mask_in.shape).astype('int64')
    fluid_index_in[j_in,i_in,k_in] = range(Npts_in)
    iC_in = fluid_index_in[j,i,k]

    # For d_dx operator
    self.OperatorDDX = scysparse.csr_matrix((Npts,Npts_in),dtype=np.float)
    iE = fluid_index_in[j,i+1,k]
    iW = fluid_index_in[j,i-1,k]
    mask_E = self.fluid_mask[j,i+1,k]
    mask_W = self.fluid_mask[j,i-1,k]
    loc = (mask_E==True)*(mask_W==True)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDX[iC[loc],iE[loc]] += 0.5*self.invdx
      self.OperatorDDX[iC[loc],iW[loc]] += -0.5*self.invdx
    loc = (mask_E==False)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDX[iC[loc],iC_in[loc]] += 1.0*self.invdx
      self.OperatorDDX[iC[loc],iW[loc]] += -1.0*self.invdx
    loc = (mask_W==False)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDX[iC[loc],iE[loc]] += 1.0*self.invdx
      self.OperatorDDX[iC[loc],iC_in[loc]] += -1.0*self.invdx

    # For d_dy operator
    self.OperatorDDY = scysparse.csr_matrix((Npts,Npts_in),dtype=np.float)
    iN = fluid_index_in[j+1,i,k]
    iS = fluid_index_in[j-1,i,k]
    mask_N = self.fluid_mask[j+1,i,k]
    mask_S = self.fluid_mask[j-1,i,k]
    loc = (mask_N==True)*(mask_S==True)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDY[iC[loc],iN[loc]] += 0.5*self.invdy
      self.OperatorDDY[iC[loc],iS[loc]] += -0.5*self.invdy
    loc = (mask_N==False)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDY[iC[loc],iC_in[loc]] += 1.0*self.invdy
      self.OperatorDDY[iC[loc],iS[loc]] += -1.0*self.invdy
    loc = (mask_S==False)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDY[iC[loc],iN[loc]] += 1.0*self.invdy
      self.OperatorDDY[iC[loc],iC_in[loc]] += -1.0*self.invdy

    # For d_dz operator
    self.OperatorDDZ = scysparse.csr_matrix((Npts,Npts_in),dtype=np.float)
    iT = fluid_index_in[j,i,k+1]
    iB = fluid_index_in[j,i,k-1]
    mask_T = self.fluid_mask[j,i,k+1]
    mask_B = self.fluid_mask[j,i,k-1]
    loc = (mask_T==True)*(mask_B==True)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDZ[iC[loc],iT[loc]] += 0.5*self.invdz
      self.OperatorDDZ[iC[loc],iB[loc]] += -0.5*self.invdz
    loc = (mask_T==False)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDZ[iC[loc],iC_in[loc]] += 1.0*self.invdz
      self.OperatorDDZ[iC[loc],iB[loc]] += -1.0*self.invdz
    loc = (mask_B==False)
    if np.sum(loc.astype('int')) > 0:
      self.OperatorDDZ[iC[loc],iT[loc]] += 1.0*self.invdz
      self.OperatorDDZ[iC[loc],iC_in[loc]] += -1.0*self.invdz

    return self.OperatorDDX, self.OperatorDDY, self.OperatorDDZ


    
  def generate_operator_d_dx2(self):
    # Generate a linear operator which dot with vector returns value of d/dx2
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iE = fluid_index[j,i+1,k]
    iW = fluid_index[j,i-1,k]
    self.OperatorDDX2 = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    
    # If the east and west nodes are inside domain, using the second order central differencing
    loc = (iE!=-1)*(iW!=-1)
    self.OperatorDDX2[iC[loc],iE[loc]] += 1.0*self.invdx2
    self.OperatorDDX2[iC[loc],iW[loc]] += 1.0*self.invdx2
    self.OperatorDDX2[iC[loc],iC[loc]] += -2.0*self.invdx2
    # For the boundary points, use 2nd order biased
    # if the east node is outside domain
    loc = (iE==-1)
    iWW = fluid_index[j[loc],i[loc]-2,k[loc]]
    self.OperatorDDX2[iC[loc],iC[loc]] += 1.0*self.invdx2
    self.OperatorDDX2[iC[loc],iWW]     += 1.0*self.invdx2
    self.OperatorDDX2[iC[loc],iW[loc]] += -2.0*self.invdx2
    # if the west node is outside domain
    loc = (iW==-1)
    iEE = fluid_index[j[loc],i[loc]+2,k[loc]]
    self.OperatorDDX2[iC[loc],iC[loc]] += 1.0*self.invdx2
    self.OperatorDDX2[iC[loc],iEE]     += 1.0*self.invdx2
    self.OperatorDDX2[iC[loc],iE[loc]] += -2.0*self.invdx2

    return self.OperatorDDX2
    
    
  def generate_operator_d_dy2(self):
    # Generate a linear operator which dot with vector returns value of d/dy2
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iN = fluid_index[j+1,i,k]
    iS = fluid_index[j-1,i,k]
    self.OperatorDDY2 = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    
    # If the north and south nodes are inside domain, using the second order central differencing
    loc = (iN!=-1)*(iS!=-1)
    self.OperatorDDY2[iC[loc],iN[loc]] += 1.0*self.invdy2
    self.OperatorDDY2[iC[loc],iS[loc]] += 1.0*self.invdy2
    self.OperatorDDY2[iC[loc],iC[loc]] += -2.0*self.invdy2
    # For the boundary point, use 2nd order biased
    # If the north node is outside domain
    loc = (iN==-1)
    iSS = fluid_index[j[loc]-2,i[loc],k[loc]]
    self.OperatorDDY2[iC[loc],iC[loc]] += 1.0*self.invdy2
    self.OperatorDDY2[iC[loc],iSS]     += 1.0*self.invdy2
    self.OperatorDDY2[iC[loc],iS[loc]] += -2.0*self.invdy2
    # If the south node is outside domain
    loc = (iS==-1)
    iNN = fluid_index[j[loc]+2,i[loc],k[loc]]
    self.OperatorDDY2[iC[loc],iC[loc]] += 1.0*self.invdy2
    self.OperatorDDY2[iC[loc],iNN]     += 1.0*self.invdy2
    self.OperatorDDY2[iC[loc],iN[loc]] += -2.0*self.invdy2

    return self.OperatorDDY2
    
    
  def generate_operator_d_dz2(self):
    # Generate a linear operator which dot with vector returns value of d/dz2
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iT = fluid_index[j,i,k+1]
    iB = fluid_index[j,i,k-1]
    self.OperatorDDZ2 = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    
    # If the top and bottom nodes are inside domain, using the second order central differencing
    loc = (iT!=-1)*(iB!=-1)
    self.OperatorDDZ2[iC[loc],iT[loc]] += 1.0*self.invdz2
    self.OperatorDDZ2[iC[loc],iB[loc]] += 1.0*self.invdz2
    self.OperatorDDZ2[iC[loc],iC[loc]] += -2.0*self.invdz2
    # For the boundary point, use 2nd order biased
    # If the top node is outside domain
    loc = (iT==-1)
    iBB = fluid_index[j[loc],i[loc],k[loc]-2]
    self.OperatorDDZ2[iC[loc],iC[loc]] += 1.0*self.invdz2
    self.OperatorDDZ2[iC[loc],iBB]     += 1.0*self.invdz2
    self.OperatorDDZ2[iC[loc],iB[loc]] += -2.0*self.invdz2
    # If the bottom node is outside domain
    loc = (iB==-1)
    iTT = fluid_index[j[loc],i[loc],k[loc]+2]
    self.OperatorDDZ2[iC[loc],iC[loc]] += 1.0*self.invdz2
    self.OperatorDDZ2[iC[loc],iTT]     += 1.0*self.invdz2
    self.OperatorDDZ2[iC[loc],iT[loc]] += -2.0*self.invdz2

    return self.OperatorDDZ2
    
    
  def generate_operator_collocated_to_staggered(self):
    # Generate the linear operators which covert the collocated values to staggered locations by averaging neighboring points
    # The staggered lcoations have 3 sets, x direction, y direction, and z direction.
    # The staggered locations are at the face centers.
    # Staggered locations are for dp/dx, dp/dy, and dp/dz values
    # The definition of staggered locations here is not same as the traditional staggered grid setup for CFD.
    # The shape of the operator matrix is [Npts_st, Npts]
    
    # Generate the fluid mask for dP_dx_staggered, dP_dy_staggered, and dP_dz_staggered.
    self.fluid_mask_x = np.zeros((self.Ny+2,self.Nx+1,self.Nz+2)).astype('bool')
    self.fluid_mask_y = np.zeros((self.Ny+1,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask_z = np.zeros((self.Ny+2,self.Nx+2,self.Nz+1)).astype('bool')
    j,i,k = np.where(self.fluid_mask==True)
    mask_E = self.fluid_mask[j,i+1,k]
    mask_N = self.fluid_mask[j+1,i,k]
    mask_T = self.fluid_mask[j,i,k+1]
    # For dp_dx (i) locations
    loc = (mask_E==True)
    self.fluid_mask_x[j[loc],i[loc],k[loc]] = True
    # For dp_dy (j) locations
    loc = (mask_N==True)
    self.fluid_mask_y[j[loc],i[loc],k[loc]] = True
    # For dp_dz (k) locations
    loc = (mask_T==True)
    self.fluid_mask_z[j[loc],i[loc],k[loc]] = True
    
    # Generate index for nodes
    # for the dp_dx_staggered
    j_x,i_x,k_x = np.where(self.fluid_mask_x==True)
    Npts_x = len(j_x)
    fluid_index_x = -np.ones(self.fluid_mask_x.shape).astype('int64')
    fluid_index_x[j_x,i_x,k_x] = range(Npts_x)
    # for the dp_dy_staggered
    j_y,i_y,k_y = np.where(self.fluid_mask_y==True)
    Npts_y = len(j_y)
    fluid_index_y = -np.ones(self.fluid_mask_y.shape).astype('int64')
    fluid_index_y[j_y,i_y,k_y] = range(Npts_y)
    # for the dp_dz_staggered
    j_z,i_z,k_z = np.where(self.fluid_mask_z==True)
    Npts_z = len(j_z)
    fluid_index_z = -np.ones(self.fluid_mask_z.shape).astype('int64')
    fluid_index_z[j_z,i_z,k_z] = range(Npts_z)
    
    # for the center nodes
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    
    # Generate the staggering linear operator for x direction
    self.staggeringOperator_x = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
    iC = fluid_index_x[j_x,i_x,k_x]
    iE = fluid_index[j_x,i_x+1,k_x]
    iW = fluid_index[j_x,i_x,k_x]
    self.staggeringOperator_x[iC,iE] += 0.5
    self.staggeringOperator_x[iC,iW] += 0.5
    
    # Generate the staggering linear operator for y direction
    self.staggeringOperator_y = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
    iC = fluid_index_y[j_y,i_y,k_y]
    iN = fluid_index[j_y+1,i_y,k_y]
    iS = fluid_index[j_y,i_y,k_y]
    self.staggeringOperator_y[iC,iN] += 0.5
    self.staggeringOperator_y[iC,iS] += 0.5

    # Generate the staggering linear operator for z direction
    self.staggeringOperator_z = scysparse.csr_matrix((Npts_z,Npts),dtype=np.float)
    iC = fluid_index_z[j_z,i_z,k_z]
    iT = fluid_index[j_z,i_z,k_z+1]
    iB = fluid_index[j_z,i_z,k_z]
    self.staggeringOperator_z[iC,iT] += 0.5
    self.staggeringOperator_z[iC,iB] += 0.5
    
    return self.staggeringOperator_x, self.staggeringOperator_y, self.staggeringOperator_z
    
  
  def generate_operator_temporal(self):
    # This fucntion is used for generating a identity matrix with dimension Npts.
    # This will be served as the linear operators for temporal derivatives.
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    self.identityOperator = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    self.identityOperator[range(Npts),range(Npts)] = 1.0
    
    return self.identityOperator


  def generate_operator_curl_staggered(self):
    # Generate the operator for calculating the curl of staggered vector fields (intended for 
    # calculating the curl of pressure gradient).

    # Generate the maks for the staggered pressure gradients. 
    self.generate_operator_collocated_to_staggered()
    # Generate flow index for the pressure gradient values.
    j_x,i_x,k_x = np.where(self.fluid_mask_x==True)
    Npts_x = len(j_x)
    fluid_index_x = -np.ones(self.fluid_mask_x.shape).astype('int64')
    fluid_index_x[j_x,i_x,k_x] = range(Npts_x)
    j_y,i_y,k_y = np.where(self.fluid_mask_y==True)
    Npts_y = len(j_y)
    fluid_index_y = -np.ones(self.fluid_mask_y.shape).astype('int64')
    fluid_index_y[j_y,i_y,k_y] = range(Npts_x,Npts_x+Npts_y)
    j_z,i_z,k_z = np.where(self.fluid_mask_z==True)
    Npts_z = len(j_z)
    fluid_index_z = -np.ones(self.fluid_mask_z.shape).astype('int64')
    fluid_index_z[j_z,i_z,k_z] = range(Npts_x+Npts_y,Npts_x+Npts_y+Npts_z)
    
    # Generate the fluid mask for the curl valus. The location should be on the center of each face.
    # For the curl along i direction
    # curl_i = dFz/dy - dFy/dz
    self.fluid_mask_curl_i = self.fluid_mask_z[1:,:,:] * self.fluid_mask_z[:-1,:,:] * self.fluid_mask_y[:,:,1:] * self.fluid_mask_y[:,:,:-1]
    j_curl_i, i_curl_i, k_curl_i = np.where(self.fluid_mask_curl_i==True) 
    Npts_curl_i = len(j_curl_i)
    index_curl_i = -np.ones(self.fluid_mask_curl_i.shape).astype('int64')
    index_curl_i[j_curl_i,i_curl_i,k_curl_i] = range(Npts_curl_i)
    self.OperatorCurl_i = scysparse.csr_matrix((Npts_curl_i,Npts_x+Npts_y+Npts_z),dtype=np.float)
    iN = fluid_index_z[j_curl_i+1,i_curl_i,k_curl_i]
    iS = fluid_index_z[j_curl_i,i_curl_i,k_curl_i]
    iT = fluid_index_y[j_curl_i,i_curl_i,k_curl_i+1]
    iB = fluid_index_y[j_curl_i,i_curl_i,k_curl_i]
    iC = index_curl_i[j_curl_i,i_curl_i,k_curl_i]
    self.OperatorCurl_i[iC,iN] += self.invdy
    self.OperatorCurl_i[iC,iS] += -self.invdy
    self.OperatorCurl_i[iC,iT] += -self.invdz
    self.OperatorCurl_i[iC,iB] += self.invdz

    # For the curl along j direction
    # curl_j = dFx/dz - dFz/dx
    self.fluid_mask_curl_j = self.fluid_mask_x[:,:,1:] * self.fluid_mask_x[:,:,:-1] * self.fluid_mask_z[:,1:,:] * self.fluid_mask_z[:,:-1,:]
    j_curl_j, i_curl_j, k_curl_j = np.where(self.fluid_mask_curl_j==True)
    Npts_curl_j = len(j_curl_j)
    index_curl_j = -np.ones(self.fluid_mask_curl_j.shape).astype('int64')
    index_curl_j[j_curl_j,i_curl_j,k_curl_j] = range(Npts_curl_j)
    self.OperatorCurl_j = scysparse.csr_matrix((Npts_curl_j,Npts_x+Npts_y+Npts_z),dtype=np.float)
    iT = fluid_index_x[j_curl_j,i_curl_j,k_curl_j+1]
    iB = fluid_index_x[j_curl_j,i_curl_j,k_curl_j]
    iE = fluid_index_z[j_curl_j,i_curl_j+1,k_curl_j]
    iW = fluid_index_z[j_curl_j,i_curl_j,k_curl_j]
    iC = index_curl_j[j_curl_j,i_curl_j,k_curl_j]
    self.OperatorCurl_j[iC,iT] += self.invdz
    self.OperatorCurl_j[iC,iB] += -self.invdz
    self.OperatorCurl_j[iC,iE] += -self.invdx
    self.OperatorCurl_j[iC,iW] += self.invdx

    # For the curl along k direction
    # curl_k = dFy/dx - dFx/dy
    self.fluid_mask_curl_k = self.fluid_mask_y[:,1:,:] * self.fluid_mask_y[:,:-1,:] * self.fluid_mask_x[1:,:,:] * self.fluid_mask_x[:-1,:,:]
    j_curl_k, i_curl_k, k_curl_k = np.where(self.fluid_mask_curl_k==True)
    Npts_curl_k = len(j_curl_k)
    index_curl_k = -np.ones(self.fluid_mask_curl_k.shape).astype('int64')
    index_curl_k[j_curl_k,i_curl_k,k_curl_k] = range(Npts_curl_k)
    self.OperatorCurl_k = scysparse.csr_matrix((Npts_curl_k,Npts_x+Npts_y+Npts_z),dtype=np.float)
    iE = fluid_index_y[j_curl_k,i_curl_k+1,k_curl_k]
    iW = fluid_index_y[j_curl_k,i_curl_k,k_curl_k]
    iN = fluid_index_x[j_curl_k+1,i_curl_k,k_curl_k]
    iS = fluid_index_x[j_curl_k,i_curl_k,k_curl_k]
    iC = index_curl_k[j_curl_k,i_curl_k,k_curl_k]
    self.OperatorCurl_k[iC,iE] += self.invdx
    self.OperatorCurl_k[iC,iW] += -self.invdx
    self.OperatorCurl_k[iC,iN] += -self.invdy
    self.OperatorCurl_k[iC,iS] += self.invdy

    # Generate a combined operator
    self.OperatorCurl = scysparse.bmat([[self.OperatorCurl_i],[self.OperatorCurl_j],[self.OperatorCurl_k]],format='csr',dtype=np.float)
    
    return self.fluid_mask_curl_i, self.fluid_mask_curl_j, self.fluid_mask_curl_k, \
           self.OperatorCurl_i, self.OperatorCurl_j, self.OperatorCurl_k, self.OperatorCurl
    
    
  def generate_gradient_operator_least_square(self):
    # This function is used for generating the gradient operator for the least square method.
    # The operator dot with pressure vector gives values of pressure gradients at staggered locations.
    
    # Generate the fluid mask for dP_dx_staggered, dP_dy_staggered, and dP_dz_staggered.
    self.fluid_mask_x = np.zeros((self.Ny+2,self.Nx+1,self.Nz+2)).astype('bool')
    self.fluid_mask_y = np.zeros((self.Ny+1,self.Nx+2,self.Nz+2)).astype('bool')
    self.fluid_mask_z = np.zeros((self.Ny+2,self.Nx+2,self.Nz+1)).astype('bool')
    j,i,k = np.where(self.fluid_mask==True)
    mask_E = self.fluid_mask[j,i+1,k]
    mask_N = self.fluid_mask[j+1,i,k]
    mask_T = self.fluid_mask[j,i,k+1]
    # For dp_dx (i) locations
    loc = (mask_E==True)
    self.fluid_mask_x[j[loc],i[loc],k[loc]] = True
    # For dp_dy (j) locations
    loc = (mask_N==True)
    self.fluid_mask_y[j[loc],i[loc],k[loc]] = True
    # For dp_dz (k) locations
    loc = (mask_T==True)
    self.fluid_mask_z[j[loc],i[loc],k[loc]] = True
    
    # Generate index for nodes
    # for the dp_dx_staggered
    j_x,i_x,k_x = np.where(self.fluid_mask_x==True)
    Npts_x = len(j_x)
    fluid_index_x = -np.ones(self.fluid_mask_x.shape).astype('int64')
    fluid_index_x[j_x,i_x,k_x] = range(Npts_x)
    # for the dp_dy_staggered
    j_y,i_y,k_y = np.where(self.fluid_mask_y==True)
    Npts_y = len(j_y)
    fluid_index_y = -np.ones(self.fluid_mask_y.shape).astype('int64')
    fluid_index_y[j_y,i_y,k_y] = range(Npts_y)
    # for the dp_dz_staggered
    j_z,i_z,k_z = np.where(self.fluid_mask_z==True)
    Npts_z = len(j_z)
    fluid_index_z = -np.ones(self.fluid_mask_z.shape).astype('int64')
    fluid_index_z[j_z,i_z,k_z] = range(Npts_z)
    
    # for the center nodes
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    
    # Generate the linear operator:
    gradientOperator_dp_dx = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
    gradientOperator_dp_dy = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
    gradientOperator_dp_dz = scysparse.csr_matrix((Npts_z,Npts),dtype=np.float)
    # for the dp_dx_staggered
    iC = fluid_index_x[j_x,i_x,k_x]
    iE = fluid_index[j_x,i_x+1,k_x]
    iW = fluid_index[j_x,i_x,k_x]
    gradientOperator_dp_dx[iC,iE] += +self.invdx
    gradientOperator_dp_dx[iC,iW] += -self.invdx
    # for the dp_dy_staggered
    iC = fluid_index_y[j_y,i_y,k_y]
    iN = fluid_index[j_y+1,i_y,k_y]
    iS = fluid_index[j_y,i_y,k_y]
    gradientOperator_dp_dy[iC,iN] += +self.invdy
    gradientOperator_dp_dy[iC,iS] += -self.invdy
    # for the dp_dz_staggered
    iC = fluid_index_z[j_z,i_z,k_z]
    iT = fluid_index[j_z,i_z,k_z+1]
    iB = fluid_index[j_z,i_z,k_z]
    gradientOperator_dp_dz[iC,iT] += +self.invdz
    gradientOperator_dp_dz[iC,iB] += -self.invdz

    gradientOperator_dp_dx_coo = gradientOperator_dp_dx.tocoo()
    gradientOperator_dp_dy_coo = gradientOperator_dp_dy.tocoo()
    gradientOperator_dp_dz_coo = gradientOperator_dp_dz.tocoo()
    dp_dx_data = gradientOperator_dp_dx_coo.data
    dp_dx_row = gradientOperator_dp_dx_coo.row
    dp_dx_col = gradientOperator_dp_dx_coo.col
    dp_dy_data = gradientOperator_dp_dy_coo.data
    dp_dy_row = gradientOperator_dp_dy_coo.row + Npts_x
    dp_dy_col = gradientOperator_dp_dy_coo.col
    dp_dz_data = gradientOperator_dp_dz_coo.data
    dp_dz_row = gradientOperator_dp_dz_coo.row + Npts_x + Npts_y
    dp_dz_col = gradientOperator_dp_dz_coo.col
    
    
    self.gradientOperatorLeastSquare = scysparse.csr_matrix((Npts_x+Npts_y+Npts_z,Npts),dtype=np.float)
    self.gradientOperatorLeastSquare[dp_dx_row,dp_dx_col] = dp_dx_data
    self.gradientOperatorLeastSquare[dp_dy_row,dp_dy_col] = dp_dy_data
    self.gradientOperatorLeastSquare[dp_dz_row,dp_dz_col] = dp_dz_data
    
    return self.gradientOperatorLeastSquare
    
  
    
  def generate_laplacian_operator_rhs_neumann(self,source_term,grad_x,grad_y,grad_z,ref_point=None):
    # Generate the Laplacian operator and the rhs given the source term and the Neumann BCs.
    
    self.Nt, self.Ny, self.Nx, self.Nz = np.shape(source_term)
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    iC = fluid_index[j,i,k]
    iE = fluid_index[j,i+1,k]
    iW = fluid_index[j,i-1,k]
    iN = fluid_index[j+1,i,k]
    iS = fluid_index[j-1,i,k]
    iT = fluid_index[j,i,k+1]
    iB = fluid_index[j,i,k-1]
    self.LaplacianOperator = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
    self.source_term = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    self.grad_x = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    self.grad_y = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    self.grad_z = np.zeros((self.Nt,self.Ny+2,self.Nx+2,self.Nz+2))
    self.RHS = np.zeros((self.Nt,Npts))
    for ct in range(self.Nt):
      self.source_term[ct,1:-1,1:-1,1:-1] = source_term[ct]
      self.grad_x[ct,1:-1,1:-1,1:-1] = grad_x[ct]
      self.grad_y[ct,1:-1,1:-1,1:-1] = grad_y[ct]
      self.grad_z[ct,1:-1,1:-1,1:-1] = grad_z[ct]
      self.RHS[ct] = self.source_term[ct,j,i,k]

    if ref_point is None:
      ref_point = [j[0]-1,i[0]-1,k[0]-1]
    
    # if the east and west nodes are inside domain
    loc = (iE!=-1)*(iW!=-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdx2
    self.LaplacianOperator[iC[loc],iE[loc]] += +1.0*self.invdx2
    self.LaplacianOperator[iC[loc],iW[loc]] += +1.0*self.invdx2
    # if the east node is outside domain
    loc = (iE==-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdx2
    self.LaplacianOperator[iC[loc],iW[loc]] += +2.0*self.invdx2
    self.RHS[:,iC[loc]] += -2.0*self.grad_x[:,j[loc],i[loc],k[loc]]*self.invdx
    # if the west node is outside domain
    loc = (iW==-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdx2
    self.LaplacianOperator[iC[loc],iE[loc]] += +2.0*self.invdx2
    self.RHS[:,iC[loc]] += +2.0*self.grad_x[:,j[loc],i[loc],k[loc]]*self.invdx
    # if the north and south nodes are inside domain
    loc = (iN!=-1)*(iS!=-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdy2
    self.LaplacianOperator[iC[loc],iN[loc]] += +1.0*self.invdy2
    self.LaplacianOperator[iC[loc],iS[loc]] += +1.0*self.invdy2
    # if the north node is outside domain
    loc = (iN==-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdy2
    self.LaplacianOperator[iC[loc],iS[loc]] += +2.0*self.invdy2
    self.RHS[:,iC[loc]] += -2.0*self.grad_y[:,j[loc],i[loc],k[loc]]*self.invdy
    # if the south node is outside domain
    loc = (iS==-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdy2
    self.LaplacianOperator[iC[loc],iN[loc]] += +2.0*self.invdy2
    self.RHS[:,iC[loc]] += +2.0*self.grad_y[:,j[loc],i[loc],k[loc]]*self.invdy
    # if the top and bottom nodes are inside domain
    loc = (iT!=-1)*(iB!=-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdz2
    self.LaplacianOperator[iC[loc],iT[loc]] += +1.0*self.invdz2
    self.LaplacianOperator[iC[loc],iB[loc]] += +1.0*self.invdz2
    # if the top node is outside domain
    loc = (iT==-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdz2
    self.LaplacianOperator[iC[loc],iB[loc]] += +2.0*self.invdz2
    self.RHS[:,iC[loc]] += -2.0*self.grad_z[:,j[loc],i[loc],k[loc]]*self.invdz
    # if the bottom node is outside domain
    loc = (iB==-1)
    self.LaplacianOperator[iC[loc],iC[loc]] += -2.0*self.invdz2
    self.LaplacianOperator[iC[loc],iT[loc]] += +2.0*self.invdz2
    self.RHS[:,iC[loc]] += +2.0*self.grad_z[:,j[loc],i[loc],k[loc]]*self.invdz
    
    j_ref = ref_point[0]
    i_ref = ref_point[1]
    k_ref = ref_point[2]
    ref_index = fluid_index[j_ref+1,i_ref+1,k_ref+1]
    self.LaplacianOperator[ref_index,:] = 0.0
    self.LaplacianOperator[ref_index,ref_index] = 1.0
    self.RHS[:,ref_index] = 0.0

    self.LaplacianOperator.eliminate_zeros()

    return self.LaplacianOperator, self.RHS
    
    
    
  def generate_laplacian_operator(self):
    # generate the Laplacian operator for the pressure Poisson solver.
    
    if self.BC == 'Dirichlet':  # If the Boundary condition is Dirichlet BC
      
      j,i,k = np.where(self.fluid_mask==True)
      Npts = len(j)
      fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
      fluid_index[j,i,k] = range(Npts)
      iC = fluid_index[j,i,k]
      iE = fluid_index[j,i+1,k]
      iW = fluid_index[j,i-1,k]
      iN = fluid_index[j+1,i,k]
      iS = fluid_index[j-1,i,k]
      iT = fluid_index[j,i,k+1]
      iB = fluid_index[j,i,k-1]
      self.LaplacianOperator = scysparse.lil_matrix((Npts,Npts),dtype=np.float)
      
      # if the node is not at boundary, apply the 7 point numerical scheme
      loc = (iE!=-1) & (iW!=-1) & (iN!=-1) & (iS!=-1) & (iT!=-1) & (iB!=-1)
      self.LaplacianOperator[iC[loc],iC[loc]] = -2.0*self.invdx2 - 2.0*self.invdy2 - 2.0*self.invdz2
      # East and west
      self.LaplacianOperator[iC[loc],iE[loc]] = +1.0*self.invdx2
      self.LaplacianOperator[iC[loc],iW[loc]] = +1.0*self.invdx2
      # North and south
      self.LaplacianOperator[iC[loc],iN[loc]] = +1.0*self.invdy2
      self.LaplacianOperator[iC[loc],iS[loc]] = +1.0*self.invdy2
      # Top and bottom
      self.LaplacianOperator[iC[loc],iT[loc]] = +1.0*self.invdz2
      self.LaplacianOperator[iC[loc],iB[loc]] = +1.0*self.invdz2
      # For the boundary points, apply the dirichlet boundary conditions.
      # if the node is outside domain
      loc = (iE==-1) | (iW==-1) | (iN==-1) | (iS==-1) | (iT==-1) | (iB==-1)
      self.LaplacianOperator[iC[loc],iC[loc]] = +1.0
      
      self.LaplacianOperator = self.LaplacianOperator.tocsc()
      
    return self.LaplacianOperator  
    
    
    
  def generate_laplacian_rhs_dirichlet(self,source_term,dirichlet_bc):
    # Generate the rhs vector of the Poisson equation 
    # The input source_term is a 3d matrix contains the source term of the poisson equation
    # The input dirichlet_bc contains the dirichlet boundary values
    
    self.source_term = np.zeros((self.Ny+2,self.Nx+2,self.Nz+2))
    self.source_term[1:-1,1:-1,1:-1] = source_term
    self.dirichlet_bc = np.zeros((self.Ny+2,self.Nx+2,self.Nz+2))
    self.dirichlet_bc[1:-1,1:-1,1:-1] = dirichlet_bc
    
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int64')
    fluid_index[j,i,k] = range(Npts)
    #iC = fluid_index[j,i,k]
    iE = fluid_index[j,i+1,k]
    iW = fluid_index[j,i-1,k]
    iN = fluid_index[j+1,i,k]
    iS = fluid_index[j-1,i,k]
    iT = fluid_index[j,i,k+1]
    iB = fluid_index[j,i,k-1]
    
    self.rhs = self.source_term[j,i,k]
    # For the boundary points, apply the dirichlet BC as the rhs
    loc = (iE==-1) | (iW==-1) | (iN==-1) | (iS==-1) | (iT==-1) | (iB==-1)
    self.rhs[fluid_index[j[loc],i[loc],k[loc]]] = self.dirichlet_bc[j[loc],i[loc],k[loc]]
    
    return self.rhs
    
  
  def generate_gradient_operator_mask_staggered(self):
    # Generate the gradient operator of 3d data.
    # The operator are generated and the mask for Gx Gy Gz 
    # This does not require the minimum width of narrow channels.

    # Generate the fluid_mask for Gx Gy and Gz
    self.fluid_mask_Gx = np.logical_and(self.fluid_mask[:,1:,:],self.fluid_mask[:,:-1,:])
    self.fluid_mask_Gy = np.logical_and(self.fluid_mask[1:,:,:],self.fluid_mask[:-1,:,:])
    self.fluid_mask_Gz = np.logical_and(self.fluid_mask[:,:,1:],self.fluid_mask[:,:,:-1])

    # Generate gradient operators.
    j,i,k = np.where(self.fluid_mask==True)
    Npts = len(j)
    fluid_index = -np.ones(self.fluid_mask.shape).astype('int')
    fluid_index[j,i,k] = range(Npts)

    # For Gx
    jx,ix,kx = np.where(self.fluid_mask_Gx==True)
    Npts_x = len(jx)
    fluid_index_x = -np.ones(self.fluid_mask_Gx.shape).astype('int')
    fluid_index_x[jx,ix,kx] = range(Npts_x)
    self.OperatorGx = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
    iE = fluid_index[jx,ix+1,kx]
    iW = fluid_index[jx,ix,kx]
    self.OperatorGx[range(Npts_x),iE] += self.invdx
    self.OperatorGx[range(Npts_x),iW] += -self.invdx

    # For Gy
    jy,iy,ky = np.where(self.fluid_mask_Gy==True)
    Npts_y = len(jy)
    fluid_index_y = -np.ones(self.fluid_mask_Gy.shape).astype('int')
    fluid_index_y[jy,iy,ky] = range(Npts_y)
    self.OperatorGy = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
    iN = fluid_index[jy+1,iy,ky]
    iS = fluid_index[jy,iy,ky]
    self.OperatorGy[range(Npts_y),iN] += self.invdy
    self.OperatorGy[range(Npts_y),iS] += -self.invdy

    # For Gz
    jz,iz,kz = np.where(self.fluid_mask_Gz==True)
    Npts_z = len(jz)
    fluid_index_z = -np.ones(self.fluid_mask_Gz.shape).astype('int')
    fluid_index_z[jz,iz,kz] = range(Npts_z)
    self.OperatorGz = scysparse.csr_matrix((Npts_z,Npts),dtype=np.float)
    iT = fluid_index[jz,iz,kz+1]
    iB = fluid_index[jz,iz,kz]
    self.OperatorGz[range(Npts_z),iT] += self.invdz
    self.OperatorGz[range(Npts_z),iB] += -self.invdz
    
    return self.OperatorGx, self.OperatorGy, self.OperatorGz, \
           self.fluid_mask_Gx[1:-1,1:-1,1:-1], self.fluid_mask_Gy[1:-1,1:-1,1:-1], self.fluid_mask_Gz[1:-1,1:-1,1:-1]






    
    
    
    
    
    
    
    
    
    
    
    
    
    
  