using Plots
using LinearAlgebra
using ITensors, ITensorMPS
using DelimitedFiles
using HDF5


function pinning_field(lx, ly, edges=false)
  N = lx*ly
  fields = zeros(N)
  if edges == true
    println("only edge field")
    fields[1], fields[2] = 1., -1.
  else
    n_column = -1
    for k in 1:lx
        n_column+=1
        for j in 1:ly
            site = j + (k-1)*ly
            println(site, "\n")
            if site % 2 == 1 && n_column % 2 == 0
                fields[site] = -1.0
            elseif site % 2 == 0 && n_column % 2 == 0
                fields[site] = 1.0
            elseif site % 2 == 1 && n_column % 2 == 1
                fields[site] = 1.0
            elseif site % 2 == 0 && n_column % 2 == 1
                fields[site] = -1.0
            end
        end
    end
  end
  return fields
end


function initialize_mps(lx, ly, sites, type)
  N = lx*ly
  if type == "afm"
    states = []
    field = pinning_field(lx, ly, false)
    for j in 1:length(field)
        if field[j] == -1.0
            push!(states, "Dn")   
        else  
            push!(states, "Up")
        end
    end
    psi = productMPS(ComplexF64, sites, states)
  else
     psi = random_mps(sites;linkdims=2)
  end
  return psi
end



function ground_state_square(Lx, Ly, J, sites)
    N = Lx*Ly
    lattice = square_lattice(Lx,Ly)
    os1 = OpSum()
    os2 = OpSum()

    for b in lattice
      os1 += J, "Sx", b.s1, "Sx", b.s2
      os1 += J, "Sy", b.s1, "Sy", b.s2
      os1 += J, "Sz", b.s1, "Sz", b.s2 
    end

    H1 = MPO(os1, sites)

    nsweeps = 10 
    maxdim = 200 
    cutoff = [1E-12] 

    psi0 = initialize_mps(lx, ly, sites, "afm")
    energy,psi = dmrg([H1],psi0;nsweeps,maxdim,cutoff)
    return energy, psi
end


function spin_correlations(ψ)
    xxcorr = correlation_matrix(ψ,"Sx","Sx")
    yycorr = correlation_matrix(ψ,"Sy","Sy")
    zzcorr = correlation_matrix(ψ,"Sz","Sz")
    C = xxcorr .+ yycorr .+ zzcorr
    return real(C)
end


function load_mps(filename)
    f = h5open(filename, "r"); 
    F = read(f, "MPS", MPS);
    close(f)
    return F 
end



function make_hamiltonian(Lx, Ly, J, sites)
    N = Lx*Ly
    lattice = square_lattice(Lx,Ly)
    os1 = OpSum()
    os2 = OpSum()
    for b in lattice
      os1 += J, "Sx", b.s1, "Sx", b.s2
      os1 += J, "Sy", b.s1, "Sy", b.s2
      os1 += J, "Sz", b.s1, "Sz", b.s2 

      os2 += 1., "Sz * Sz", b.s1
    end
    H1 = MPO(os1, sites)
    H2 = MPO(os2, sites)
    return H1 #+ H2
end


function mean_internal_energy(psi, lx, ly, J)
    sites = siteinds(psi)
    h = make_hamiltonian(lx, ly, J, sites)
    mean_h = inner(psi, h, psi)
    return mean_h
end


function mean_sz(psi)
  return expect(psi, "Sz") 
end


function staggered_magnetization(psi, lx, ly)
  mean_mag = sum(abs.(mean_sz(psi)))
  return mean_mag
end


##SQUARE LATTICE HEISENBERG SPIN HALF INITIALIZTION;
lx, ly = 10, 10
j = 1.
sites = siteinds("S=1/2", lx*ly; conserve_qns = false)




###MEAN GROUND STATE ENERGY
psi0 = load_mps("C:/Users/aroge/Desktop/KITP/square_heinsenberg_half/psi0.h5")
psi2 = load_mps("C:/Users/aroge/Desktop/KITP/square_heinsenberg_half/psi_v2.h5")

res0 = mean_internal_energy(psi0, lx, ly, j)
res2 = mean_internal_energy(psi2, lx, ly, j)

println("initial state E0/bound = ", res0/180, "\n")
println("optimized, good initialization: E0/bound = ", res2/180, "\n")

###STAGGERED MAGNETZIATION 
mag1 = staggered_magnetization(psi0, lx, ly)
mag2 = staggered_magnetization(psi2, lx, ly)
println("mag initial = ", mag1/(lx*ly), "\n")
println("mag final = ", mag2/(lx*ly), "\n")



###DOWNLOAD INITIAL STATE
# psi0 = initialize_mps(lx, ly, sites, "afm")
# fo = h5open("C:/Users/aroge/Desktop/KITP/square_heinsenberg_half/psi0.h5", "w")
# write(fo, "MPS", psi0)
# close(fo)

# Cs0 = spin_correlations(psi0)
# writedlm("C:/Users/aroge/Desktop/KITP/square_heinsenberg_half/spin_correl_ini.txt", Cs0)


####GENERATE GROUND STATE FROM DMRG
# e, ψ = ground_state_square(lx, ly, j, sites)
# println("ground state:", e/(lx*ly))
# ##download mps
# fo = h5open("C:/Users/aroge/Desktop/KITP/square_heinsenberg_half/psi_v2.h5", "w")
# write(fo, "MPS",  ψ)
# close(fo)

# # ##dowload correl
# Cs = spin_correlations(ψ)
# writedlm("C:/Users/aroge/Desktop/KITP/square_heinsenberg_half/spin_correl_v2.txt", Cs)




