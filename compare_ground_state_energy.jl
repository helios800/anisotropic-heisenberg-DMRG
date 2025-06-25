using Plots
using LinearAlgebra
using ITensors, ITensorMPS
using DelimitedFiles
using HDF5


###load MPS from DMRG results
function load_mps(path, D, lambd, theta, field)
    filename = "DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(field).h5"
    fullpath = joinpath(path, filename)
    f = h5open(fullpath, "r"); 
    F = read(f, "MPS", MPS);
    close(f)
    return F 
end


function load_mps_v2(filename, path)
    fullpath = joinpath(path, filename)
    f = h5open(fullpath, "r"); 
    F = read(f, "MPS", MPS);
    close(f)
    return F 
end


### internal energy hamiltonian
function make_hamiltonian(Lx, Ly, J, lambd, theta, D, sites)
    N = Lx*Ly
    lattice = triangular_lattice(Lx,Ly; yperiodic = true)
    os1 = OpSum()
    os2 = OpSum()
    os3 = OpSum()
    ###lucile!
    for b in lattice
      ##H1
      os1 += J*lambd*cos(theta), "Sx", b.s1, "Sx", b.s2
      os1 += J*lambd*cos(theta), "Sy", b.s1, "Sy", b.s2
      os1 += J*cos(theta), "Sz", b.s1, "Sz", b.s2 
      ##H2
      #terme (a) du developpement
      os2 +=  J*lambd^2*sin(theta), "Sx", b.s1, "Sx", b.s2, "Sx", b.s1, "Sx", b.s2
      os2 +=  J*lambd^2*sin(theta), "Sx", b.s1, "Sx", b.s2, "Sy", b.s1, "Sy", b.s2
      os2 +=  J*lambd^2*sin(theta), "Sy", b.s1, "Sy", b.s2, "Sx", b.s1, "Sx", b.s2
      os2 +=  J*lambd^2*sin(theta), "Sy", b.s1, "Sy", b.s2, "Sy", b.s1, "Sy", b.s2
      #terme (b) du developpement
      os2 +=  J*lambd*sin(theta), "Sx", b.s1, "Sx", b.s2, "Sz", b.s1, "Sz", b.s2
      os2 +=  J*lambd*sin(theta), "Sy", b.s1, "Sy", b.s2, "Sz", b.s1, "Sz", b.s2
      os2 +=  J*lambd*sin(theta), "Sz", b.s1, "Sz", b.s2, "Sx", b.s1, "Sx", b.s2
      os2 +=  J*lambd*sin(theta), "Sz", b.s1, "Sz", b.s2, "Sy", b.s1, "Sy", b.s2
      #terme (c) du developpement
      os2 +=  J*sin(theta), "Sz", b.s1, "Sz", b.s2, "Sz", b.s1, "Sz", b.s2
      ##H_D
      os3 +=  D, "Sz", b.s1, "Sz", b.s1
    end
    H1 = MPO(os1, sites)
    H2 = MPO(os2, sites)
    HD = MPO(os3, sites)
    return H1 + H2 + HD
end


function mean_internal_energy(psi, lx, ly, J, lambd, theta, D)
    sites = siteinds(psi)
    h = make_hamiltonian(lx, ly, J, lambd, theta, D, sites)
    mean_h = inner(psi, h, psi)
    return mean_h
end



J = 1.0
theta = 1.1780972450961724
D = -0.2
lambd = 0.3
ext_field = 0.0
lx, ly = 10, 6


path = "C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/lambd_0.4/"
filename = "DMRG_afm_lambd_0.4.h5"
psi2 = load_mps_v2(filename, path)
mean_ho = mean_internal_energy(psi2, lx, ly, J, lambd, theta, D)
println("mean H0, AFM  initial state = ", real(mean_ho)/(lx*ly))

filename = "DMRG_wannier_lambd_0.4.h5"
psi2 = load_mps_v2(filename, path)
mean_ho = mean_internal_energy(psi2, lx, ly, J, lambd, theta, D)
println("mean H0, Wannier  initial state = ", real(mean_ho)/(lx*ly))

filename = "DMRG_stripes_lambd_0.4.h5"
psi2 = load_mps_v2(filename, path)
mean_ho = mean_internal_energy(psi2, lx, ly, J, lambd, theta, D)
println("mean H0, stripes initial state = ", real(mean_ho)/(lx*ly))





# ###STARTING RANDOM WANIER STATE
# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "psi0_wanier.h5"
# psi0 = load_mps_v2(filename, path)
# mean_ho = mean_internal_energy(psi0, lx, ly, J, lambd, theta, D)
# println("mean H0, wanier initial state = ", real(mean_ho)/(lx*ly))


# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field)_ini_wanier.h5"
# psi1 = load_mps_v2(filename, path)
# mean_ho = mean_internal_energy(psi1, lx, ly, J, lambd, theta, D)
# println("mean H0, wanier final state = ", real(mean_ho)/(lx*ly))


# ###STARTING AFM STATE
# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "psi0_afm.h5"
# psi2 = load_mps_v2(filename, path)
# mean_ho = mean_internal_energy(psi2, lx, ly, J, lambd, theta, D)
# println("mean H0, AFM  initial state = ", real(mean_ho)/(lx*ly))


# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field)_ini_afm.h5"
# psi3 = load_mps_v2(filename, path)
# mean_ho = mean_internal_energy(psi3, lx, ly, J, lambd, theta, D)
# println("mean H0, AFM final state = ", real(mean_ho)/(lx*ly))







### edge field, 0.12 external field QQ-AF test. (no AFM order)
# J = 1.0
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.3 #0.12
# lx, ly = 10, 6

# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# psi = load_mps(path, D, lambd, theta, field)
# mean_ho = mean_internal_energy(psi, lx, ly, J, lambd, theta, D)
# println("mean H0, edge field (QQ-AF) = ", real(mean_ho)/(lx*ly))


# ### bulk field, 0.075 field QQ-AF test. (AFM order)
# J = 1.0
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.075
# lx, ly = 10, 6

# path = "C:/Users/aroge/Desktop/KITP/FSS_field/lx_10_ly_6/"
# psi = load_mps(path, D, lambd, theta, field)
# mean_ho = mean_internal_energy(psi, lx, ly, J, lambd, theta, D)
# println("mean H0, bulk field (QQ-AF), order = ", real(mean_ho)/(lx*ly))



# ### bulk field, 0.05 field QQ-AF test. (no AFM order)
# J = 1.0
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.05
# lx, ly = 10, 6

# path = "C:/Users/aroge/Desktop/KITP/FSS_field/lx_10_ly_6/"
# psi = load_mps(path, D, lambd, theta, field)
# mean_ho = mean_internal_energy(psi, lx, ly, J, lambd, theta, D)
# println("mean H0, bulk field (QQ-AF), no order = ", real(mean_ho)/(lx*ly))



# ###no field, QQ-AF test. (no AFM order)
# J = 1.0
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.0
# lx, ly = 10, 6

# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# psi = load_mps(path, D, lambd, theta, field)
# mean_ho = mean_internal_energy(psi, lx, ly, J, lambd, theta, D)
# println("mean H0, no field (QQ-AF) = ", real(mean_ho)/(lx*ly))


