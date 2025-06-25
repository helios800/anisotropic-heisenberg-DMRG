  
using Plots
using LinearAlgebra
using ITensors, ITensorMPS
using DelimitedFiles
using HDF5



function pinning_field(n_spins)
  fields = 1.0*ones(n_spins)
  count_down = 0
  count_up = 2
  for k in 1:n_spins
      #println("k=", k, "\n")
      if count_up < 4 && count_up > 0
          #println("1", "\n")
          count_up += 1
      elseif count_up == 4 
          #println("2", "\n")
          fields[k] = -1.0
          count_up = 0
          count_down += 1
      elseif count_down < 2 && count_down > 0
         # println("3", "\n")
          fields[k] = -1.
          count_down += 1

      elseif count_down == 2
          #println("4", "\n")
          fields[k] = 1.0
          count_down = 0
          count_up += 1
      else
          count_up += 1
      end
  end
  return fields
end





function entropy_von_neumann(psi::MPS, b::Int)
  s = siteinds(psi)  
  orthogonalize!(psi, b)
  _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
  SvN = 0.0
  for n in 1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  return SvN
end



function complex_correlation_matrix(ψ, sites, opname1, opname2)
    N = length(sites)
    C = zeros(ComplexF64, N, N)

    for i in 1:N
        for j in 1:N
            op_list = [op("Id", sites[k]) for k in 1:N]
            op_list[i] = op(opname1, sites[i])
            op_list[j] = op(opname2, sites[j])
            Oij = MPO(op_list)
            C[i,j] = inner(ψ, Oij, ψ)
        end
    end
    return real(C)
end


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

    # cc = complex_correlation_matrix(ψ, sites, "Qc", "Qc")
    # cd = complex_correlation_matrix(ψ, sites, "Qd", "Qd")
    # ce = complex_correlation_matrix(ψ, sites, "Qe", "Qe")
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


function triangular_lattice_coords(n1_max::Int, n2_max::Int, a::Float64=1.0)
    a1 = (2π/a) * [1.0, -1/sqrt(3)]
    a2 = (4π/(a*sqrt(3))) * [0.0, 1.0]
    coords = []
    for n1 in 0:n1_max-1
        for n2 in 0:n2_max-1
            r = n1 .* a1 .+ n2 .* a2
            push!(coords, r)
        end
    end
    return coords
end


function spin_sites_afm(Lx, Ly)
  s = 1
  site_list = []
  for i in 1:Ly
    for j in 1:Lx
      s = (j-1)*Ly + i    
      if j % 3 == 1 && i % 3 == 1
        push!(site_list, s)
      elseif j % 3 == 2 && i % 3 == 0
        push!(site_list, s)
      elseif j % 3 == 0 && i % 3 == 2
        push!(site_list, s)
      end
    end
  end
  return sort(site_list)
end


function ground_state_triangular(Lx, Ly, J, lambd, theta, D, ext_field, sites)
    N = Lx*Ly
    lattice = triangular_lattice(Lx,Ly; yperiodic = true)
    os1 = OpSum()
    os2 = OpSum()
    os3 = OpSum()
    os4 = OpSum()

    # for b in lattice
    #   os1 += J, "Sx", b.s1, "Sx", b.s2
    #   os1 += J, "Sy", b.s1, "Sy", b.s2
    #   os1 += J, "Sz", b.s1, "Sz", b.s2
    # end

     # #milla
    # for b in lattice
    #   os1 += cos(theta), "Sx", b.s1, "Sx", b.s2
    #   os1 += cos(theta), "Sy", b.s1, "Sy", b.s2
    #   os1 += cos(theta), "Sz", b.s1, "Sz", b.s2
  
    #   os2 +=  sin(theta), "Sx", b.s1, "Sx", b.s2, "Sx", b.s1, "Sx", b.s2
    #   os2 +=  sin(theta), "Sy", b.s1, "Sy", b.s2, "Sy", b.s1, "Sy", b.s2
    #   os2 +=  sin(theta), "Sz", b.s1, "Sz", b.s2, "Sz", b.s1, "Sz", b.s2
    # end

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
      os3 +=  D, "Sz", b.s1, "Sz", b.s1  #, "Sy", b.s1, "Sy", b.s2
    end

    ## symmetry breaking by pinning field: 
   sites_with_spins = spin_sites_afm(Lx, Ly)
   n_spins = length(sites_with_spins)
   local_field = pinning_field(n_spins)
   for l in 1:n_spins
      spin_site = sites_with_spins[l]
      constant = ext_field*local_field[l]
      println("sites:", spin_site, "field:", constant, "\n")
      os4 += constant, "Sz", spin_site
    end


    H1 = MPO(os1, sites)
    H2 = MPO(os2, sites)
    HD = MPO(os3, sites)
    Hs = MPO(os4, sites)

    nsweeps = 10 #10
    maxdim = 200 #500
    cutoff = [1E-12] 

    psi0 = random_mps(sites;linkdims=2)
    #state = ["Dn" for i in 1:N]
    #psi0 = productMPS(sites, state)
    energy,psi = dmrg([H1,H2,HD,Hs],psi0;nsweeps,maxdim,cutoff) #no Hs here
    return energy, psi
end





lx, ly = 10, 9 #10, 6
N = lx*ly
##cf lucile. fig.6
j = 1.
lambd = 0.1 #0.0 #0.25 #1.
theta = 3*pi/8  #-pi/4 #0.5*pi #-3/4*pi 
D = -0.2 #-0.12 #-0.21  #2. #4. #0.
#ext_field = 0.1
sites = siteinds("S=1", lx*ly; conserve_qns = false)

for ext_field in [0.075] #[0.05, 0.075, 0.1]
  e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
  println("field:", ext_field)
  println("ground state:", e/(lx*ly), "\n")
  ### download mps
  fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field).h5", "w")
  write(fo, "MPS", ψ)
  close(fo)

  ##download correl
  C_s = spin_correlations(ψ)
  C_q = quadrupolar_correlations(ψ, sites)
  writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/spin_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field).txt", C_s)
  writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/quadrupolar_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field).txt", C_q)
end


###load mps back
# f = h5open("DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta).h5","r");  ##MPS result of DMRG
# F = read(f, "MPS", MPS);
# close(f)












# C_s = spin_correlations(F)
# println(size(C_s))





#writedlm("entropies_theta_$(theta)_lambd_$(lambd)_D_$(D).txt", entropies)


# correlations = C_q[:, 1] 
# #correlations = C_s[:,1]
# println(correlations)


# a2 = (4π/(sqrt(3)))

# xarr = [] 
# yarr = [] 
# labels = []
# positions = triangular_lattice_coords(lx, ly)

# for (i, pos) in enumerate(positions)
#     push!(xarr, pos[1])
#     push!(yarr, pos[2])
#     push!(labels, "$(i)")
# end

# scatter(xarr, yarr, marker_z = correlations, c=:viridis, ms=10,
#         xlabel="x", ylabel="y", title="⟨Sz₁ Szⱼ⟩ correlations",
#         colorbar=true)

# # Add text annotations
# for (i, (x, y)) in enumerate(zip(xarr, yarr))
#     annotate!(x, y, text(labels[i], :center, 8, :black))
# end
# # Show the plot
# display(current())

