
using LinearAlgebra
using ITensors, ITensorMPS
using DelimitedFiles
using HDF5


##CORRELATION MATRICES
function spin_correlations(ψ)
    xxcorr = correlation_matrix(ψ,"Sx","Sx")
    yycorr = correlation_matrix(ψ,"Sy","Sy")
    zzcorr = correlation_matrix(ψ,"Sz","Sz")
    C = xxcorr .+ yycorr .+ zzcorr
    return real(C)
end

function sz_correl(psi)
    corr = correlation_matrix(psi, "Sz", "Sz")
    return corr 
end

function ortho_correl(psi)
    corr = correlation_matrix(psi,"S+","S-")
    return corr
end


##COMPUTE GROUND STATE
function ground_state_triangular(Lx, Ly, J1, J2, sites)
    N = Lx*Ly
    lattice = triangular_lattice(Lx,Ly; yperiodic = true)
    os1 = OpSum()
    os2 = OpSum()
    for b in lattice
      ##H1
      os1 += J1, "Sx", b.s1, "Sx", b.s2
      os1 += J1, "Sy", b.s1, "Sy", b.s2
      ##H2
      os2 += J2, "Sz", b.s1, "Sz", b.s2
    end
    H1 = MPO(os1, sites)
    H2 = MPO(os2, sites)
    nsweeps = 10 
    maxdim = 400 
    cutoff = [1E-12] 

    psi0 = random_mps(sites;linkdims=2)
    energy,psi = dmrg([H1,H2],psi0;nsweeps,maxdim,cutoff)
    return energy, psi
end


###GENERATE RESULTS
lx, ly = 10, 6
N = lx*ly
j1 = 1.
#j2 = 2.

for j2 in [0.02, 0.1, 0.3, 0.5, 0.7, 1.2, 1.4, 1.6, 1.8]  #[0.05, 1., 2., 3., 4., 5., 6., 8.]
    println("DELTA = ", j2/j1, "\n")
    sites = siteinds("S=1/2", lx*ly; conserve_qns = false)
    e, psi = ground_state_triangular(lx, ly, j1, j2, sites)
    println("ground state energy = ", e/(N), "\n")

    C_s = real(spin_correlations(psi))
    Cz = real(sz_correl(psi))
    C_ortho = real(ortho_correl(psi))

    writedlm("C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/spin_correl_delta_$(j2/j1).txt", C_s)
    writedlm("C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/Sz_correl_delta_$(j2/j1).txt", Cz)
    writedlm("C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/ortho_correl_delta_$(j2/j1).txt", C_ortho)

    fo = h5open("C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/psi_delta_$(j2/j1).h5", "w")
    write(fo, "MPS", psi)
    close(fo)
end