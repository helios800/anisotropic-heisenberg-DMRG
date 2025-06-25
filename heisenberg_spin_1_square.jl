using Plots
using LinearAlgebra
using ITensors, ITensorMPS
using DelimitedFiles
using HDF5


function ITensors.op(::OpName"Qzz", ::SiteType"S=1", s::Index)
  Sz2 = op("Sz2", s)
  I = op("Id", s)
  return sqrt(2)*(Sz2 - 2*I/3) 
end


function ITensors.op(::OpName"Qa", ::SiteType"S=1", s::Index)
    Sx2 = op("Sx2", s)
    Sy2 = op("Sy2", s)
    return Sx2 - Sy2 
  end
  
  function ITensors.op(::OpName"Qb", ::SiteType"S=1", s::Index)
    Sx2 = op("Sx2", s)
    Sy2 = op("Sy2", s)
    Sz2 = op("Sz2", s)
    return (2 * Sz2 - Sx2 - Sy2)/sqrt(3) 
  end
  
  function ITensors.op(::OpName"Qc", ::SiteType"S=1", s::Index)
    # Get the spin operators
    sx = op("Sx * Sy", s)
    sy = op("Sy * Sx", s)
    return sx + sy
  end
  
  function ITensors.op(::OpName"Qd", ::SiteType"S=1", s::Index)
    # Get the spin operators
    sz = op("Sz * Sy", s)
    sy = op("Sy * Sz", s)
    return sz + sy
  end
          
  function ITensors.op(::OpName"Qe", ::SiteType"S=1", s::Index)
    sx = op("Sx * Sz", s)
    sz = op("Sz * Sx", s)
    return  sx + sz
  end        
     

function quadrupolar_correlations(ψ, sites)
    ca = correlation_matrix(ψ,"Qa","Qa")
    cb = correlation_matrix(ψ,"Qb","Qb")
    cc = correlation_matrix(complex(ψ),"Qc","Qc")
    cd = correlation_matrix(complex(ψ),"Qd","Qd")
    ce = correlation_matrix(complex(ψ),"Qe","Qe")
    C = ca .+ cb .+ cc .+ cd .+ ce
    return real(C)
end


function spin_correlations(ψ)
    xxcorr = correlation_matrix(ψ,"Sx","Sx")
    yycorr = correlation_matrix(ψ,"Sy","Sy")
    zzcorr = correlation_matrix(ψ,"Sz","Sz")
    C = xxcorr .+ yycorr .+ zzcorr
    return real(C)
end




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





##SQUARE LATTICE HEISENBERG SPIN HALF INITIALIZTION;
lx, ly = 10, 10
j = 1.
D = 0.2

sites = siteinds("S=1", lx*ly; conserve_qns = false)

e, ψ = ground_state_square(lx, ly, j, D, sites)
fo = h5open("C:/Users/aroge/Desktop/KITP/square_heisenberg_1/DMRG_lambda_$(D).h5", "w")
write(fo, "MPS", ψ)
close(fo)

C_s = spin_correlations(ψ)
C_q = quadrupolar_correlations( ψ, sites)
writedlm("C:/Users/aroge/Desktop/KITP/square_heisenberg_1/spin_lambda_$(lambd).txt", C_s)
writedlm("C:/Users/aroge/Desktop/KITP/square_heisenberg_1/quadrupolar_lambda_$(lambd).txt", C_q)

