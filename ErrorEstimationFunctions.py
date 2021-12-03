import numpy as np
import sys
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
from timeit import default_timer

sys.path.insert(0, '../')
from NumericalLinearOperators import LinearOperatorGeneration
from EvaluatePressureGradient import NavierStokesMomentum

def weighted_standard_deviation_3D(Val,mask,sigma_r=2,sigma_t=1):
  """
  # Calculalte a weighted std (Bivariate Gaussian kernel) of the input Val.
  # The input should be a 4d np.array with shape (Nt,Ny,Nx,Nz)
  # The input mask is the static mask in the shape of (Ny,Nx,Nz)
  # The sigma_r is the sigma of the Bivariate Gaussian in space (in terms of grid).
  # The sigma_t is the sigma of the Bivariate Gaussian in time (in terms of grid).
  # This is mainly used for calculating the weighted STD of the predicted pgrad errors.
  Inputs:
    Val is the 3d array of value field based on which the WSTD are calculated. 
    mask is the 3d array of binary mask
    sigma_r is the spatail correlation length
    sigma_t is the temporal correlation length
  Returns:
    Val_wstd: the 3d array contianing the 3d field of wstd
  """
  Nt,Ny,Nx,Nz = np.shape(Val)

  sigma_r = int(sigma_r)
  sigma_r = np.max([1,sigma_r])
  sigma_t = int(sigma_t)
  sigma_t = np.max([1,sigma_t])

  # Generate a brick containing the weights according to the input of sigma_r and sigma_t
  birck_r_space = 2*sigma_r # Spatial size of brick is 2*sigma_r
  brick_r_time = sigma_t # Temporal is only 1*sigma_t as the error may be less correlated in time.
  Weights_brick = np.zeros((brick_r_time*2 + 1, birck_r_space*2 + 1, birck_r_space*2 + 1, birck_r_space*2 + 1))
  xn_brick = np.arange(birck_r_space*2 + 1)
  yn_brick = np.arange(birck_r_space*2 + 1)
  zn_brick = np.arange(birck_r_space*2 + 1)
  Xn_brick, Yn_brick, Zn_brick = np.meshgrid(xn_brick,yn_brick,zn_brick)
  Xn_brick = np.tile(Xn_brick,(brick_r_time*2+1,1,1,1))
  Yn_brick = np.tile(Yn_brick,(brick_r_time*2+1,1,1,1))
  Zn_brick = np.tile(Zn_brick,(brick_r_time*2+1,1,1,1))
  Tn_birck = np.zeros(Weights_brick.shape)
  for ct in range(brick_r_time*2 + 1):
    Tn_birck[ct,:,:,:] = ct
  # Make the center to be zero xyzt 
  Xn_brick -= Xn_brick[brick_r_time,birck_r_space,birck_r_space,birck_r_space]
  Yn_brick -= Yn_brick[brick_r_time,birck_r_space,birck_r_space,birck_r_space]
  Zn_brick -= Zn_brick[brick_r_time,birck_r_space,birck_r_space,birck_r_space]
  Rn_brick = (Xn_brick**2 + Yn_brick**2 + Zn_brick**2)**0.5
  Tn_birck -= Tn_birck[brick_r_time,birck_r_space,birck_r_space,birck_r_space]
  Weights_brick = np.exp(-0.5*(Rn_brick/sigma_r)**2) * np.exp(-0.5*(Tn_birck/sigma_t)**2)

  # Apply this Weight brick to each point in both time and space.
  Val_wstd = np.zeros(Val.shape)
  mask = np.tile(mask,(Nt,1,1,1))
  t_pts,j_pts,i_pts,k_pts = np.where(mask == True)
  Npts = len(t_pts)

  Val_padding = np.zeros((Nt+brick_r_time*2, Ny+birck_r_space*2, Nx+birck_r_space*2, Nz+birck_r_space*2))
  Val_padding[brick_r_time:-brick_r_time, birck_r_space:-birck_r_space, birck_r_space:-birck_r_space, birck_r_space:-birck_r_space] = Val
  mask_padding = np.zeros((Nt+brick_r_time*2, Ny+birck_r_space*2, Nx+birck_r_space*2, Nz+birck_r_space*2))
  mask_padding[brick_r_time:-brick_r_time, birck_r_space:-birck_r_space, birck_r_space:-birck_r_space, birck_r_space:-birck_r_space] = mask.astype('float')
  for ct in range(Npts):
    t = t_pts[ct]
    j = j_pts[ct]
    i = i_pts[ct]
    k = k_pts[ct]
    Val_brick = Val_padding[t:t+brick_r_time*2+1, j:j+birck_r_space*2+1, i:i+birck_r_space*2+1, k:k+birck_r_space*2+1]
    mask_brick = mask_padding[t:t+brick_r_time*2+1, j:j+birck_r_space*2+1, i:i+birck_r_space*2+1, k:k+birck_r_space*2+1]
    Val_wstd[t,j,i,k] = (np.sum(Val_brick**2 * mask_brick * Weights_brick) / np.sum(mask_brick * Weights_brick))**0.5

  return Val_wstd


def velocity_error_estimation_from_divergence(Xn,Yn,Zn,fluid_maskn,Un,Vn,Wn):
  """
  # Inputs: Grid setup for the 3d field, Xn, Yn, Zn, fluid_maskn in shape of (Ny,Nx,Nz)
  # Velocity field Un, Vn, Wn in shape of (Nt,Ny,Nx,Nz)
  # Outputs: error esitmation of Un Vn & Wn and the spurious divergence
  Inputs:
    Xn,Yn,Zn: 3d mesh grids
    fluid_maskn: binary mask of flow field.
    Un,vn,Wn: 3d velocity fields given as 4d array with the shape (Nt,Ny,Nx,Nz)
  Returns:
    Un_error,Vn_error,Wn_error: velocity error fields given as 4d array with the shape of (Nt,Ny,Nx,Nz)
    divergence_field: divergence fields of the velocity fields given as 4d array with the shape of (Nt,Ny,Nx,Nz)
  """

  Nt, Ny, Nx, Nz = np.shape(Un)
  dx = Xn[0,1,0] - Xn[0,0,0]
  dy = Yn[1,0,0] - Yn[0,0,0]
  dz = Zn[0,0,1] - Zn[0,0,0]

  # Put the velocity values into stack of 1d vectors. 
  j,i,k = np.where(fluid_maskn == True)
  Npts = len(j)
  velocity_vector_stack = np.zeros((Nt,3*Npts))
  for ct in range(Nt):
    velocity_vector_stack[ct,:Npts] = Un[ct,j,i,k]
    velocity_vector_stack[ct,Npts:2*Npts] = Vn[ct,j,i,k]
    velocity_vector_stack[ct,2*Npts:] = Wn[ct,j,i,k]

  # Generate the divergence operator
  linear_operator_generator = LinearOperatorGeneration(Xn,Yn,Zn,fluid_maskn)
  d_dx = linear_operator_generator.generate_operator_d_dx()
  d_dy = linear_operator_generator.generate_operator_d_dy()
  d_dz = linear_operator_generator.generate_operator_d_dz()

  # Generate the divergence operator
  divergence_operator = scysparse.bmat([[d_dx,d_dy,d_dz]],format='csr',dtype=np.float)

  # Evaluate the spurious divergence of the input velocity field.
  divergence_stack = np.zeros((Nt,Npts))
  divergence_field = np.zeros((Nt,Ny,Nx,Nz))
  for ct in range(Nt):
    divergence_stack[ct] = divergence_operator.dot(velocity_vector_stack[ct])
    divergence_field[ct,j,i,k] = divergence_stack[ct]

  # Solve for the least-norm solution from the under-determined system
  LHS = divergence_operator * divergence_operator.transpose()
  RHS = splinalg.spsolve(LHS, divergence_stack.transpose())
  velocity_error = divergence_operator.transpose() * RHS
  velocity_error = velocity_error.transpose()

  Un_error = np.zeros((Nt,Ny,Nx,Nz))
  Vn_error = np.zeros((Nt,Ny,Nx,Nz))
  Wn_error = np.zeros((Nt,Ny,Nx,Nz))

  for ct in range(Nt):
    Un_error[ct,j,i,k] = velocity_error[ct,:Npts]
    Vn_error[ct,j,i,k] = velocity_error[ct,Npts:2*Npts]
    Wn_error[ct,j,i,k] = velocity_error[ct,2*Npts:]

  return Un_error, Vn_error, Wn_error, divergence_field


def pressure_gradient_error_estimation_from_divergence(Xn,Yn,Zn,fluid_maskn,Un,Vn,Wn,nu,rho,dt,gravity_direction=None):
  """
  # Evaluate the velocity error from velocity divergence, 
  # then the pressure gradient error is estimated based on that.
  Inputs:
    Xn,Yn,Zn: 3d mesh grids
    fluid_maskn: binary mask of flow field.
    Un,vn,Wn: 3d velocity fields given as 4d array with the shape (Nt,Ny,Nx,Nz)
    nu: kinematic viscosity
    rho: density
    dt: time difference between snapshots
    gravity_direction: direction of the gravity acceleration. None means not considered.
  Returns:
    dP_dx_staggered, dP_dy_staggered, dP_dz_staggered: 3d pressure gradient fields given as 4d array. Located on staggered grid locaitons.
    dP_dx_staggered_error, dP_dy_staggered_error, dP_dz_staggered_error: estiamted pgrad errors in the same form as pgrads.
  """

  Un_error, Vn_error, Wn_error, divergence_field = velocity_error_estimation_from_divergence(Xn,Yn,Zn,fluid_maskn,Un,Vn,Wn)

  Un_corrected = Un - Un_error
  Vn_corrected = Vn - Vn_error
  Wn_corrected = Wn - Wn_error

  pressure_gradient_calculator = NavierStokesMomentum(Xn,Yn,Zn,Un,Vn,Wn,fluid_maskn,nu,rho,dt,gravity_direction=gravity_direction)
  dP_dx_staggered, dP_dy_staggered, dP_dz_staggered, fluid_mask_dp_dx, fluid_mask_dp_dy, fluid_mask_dp_dz = \
    pressure_gradient_calculator.eval_pressure_gradient_staggered()

  pressure_gradient_calculator = NavierStokesMomentum(Xn,Yn,Zn,Un_corrected,Vn_corrected,Wn_corrected,fluid_maskn,nu,rho,dt,gravity_direction=gravity_direction)
  dP_dx_staggered_corrected, dP_dy_staggered_corrected, dP_dz_staggered_corrected, fluid_mask_dp_dx, fluid_mask_dp_dy, fluid_mask_dp_dz = \
    pressure_gradient_calculator.eval_pressure_gradient_staggered()

  dP_dx_staggered_error = dP_dx_staggered - dP_dx_staggered_corrected
  dP_dy_staggered_error = dP_dy_staggered - dP_dy_staggered_corrected
  dP_dz_staggered_error = dP_dz_staggered - dP_dz_staggered_corrected

  return dP_dx_staggered, dP_dy_staggered, dP_dz_staggered, dP_dx_staggered_error, dP_dy_staggered_error, dP_dz_staggered_error










