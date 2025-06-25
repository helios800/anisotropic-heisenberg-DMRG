
using Plots
using LinearAlgebra
using ITensors, ITensorMPS
using DelimitedFiles
using HDF5



function pinning_field(n_spins, edges=false)

  if edges == true
    println("only edge field")
    fields = zeros(n_spins)
    fields[1], fields[2] = 1., 1.

  else
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





function initialize_mps(lx, ly, sites, type)
  N = lx*ly
  states = []
  spin_sites = spin_sites_afm(lx, ly)

  if type=="afm"
    index_field = 0
    spin_sites = spin_sites_afm(lx, ly)
    field = pinning_field(length(spin_sites), false)
    for k in 1:N 
      if k in spin_sites
        index_field += 1
        if field[index_field] == -1.
          push!(states, "Dn")
        else
          push!(states, "Up")
        end
      else
        push!(states, "Z0")
      end
    end
    psi = productMPS(ComplexF64, sites, states)

  elseif type == "wanier"
    if lx == 6
      field = [-1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,
        1.,  0.,  0., -1.,  0., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,
        1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.]
    elseif lx == 10
      field = [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  0.,
                -1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
                  1.,  0.,  0., -1.,  0., -1.,  0.,  0., -1.,  0., -1.,  0.,  0.,
                -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0., -1.,  0.,  0.,
                -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]
    elseif lx == 16
      field = [-1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
        1.,  0.,  0., -1.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,
       -1.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  1.,  0., -1.,  0.,  0.,
        1.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0., -1.,  0.,  0.,
       -1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
       -1.,  0., -1.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
        0.,  0.,  1.,  0.,  0., -1.,  0., -1.,  0.,  0., -1.,  0.,  0.,
        0.,  0.,  1.,  0.,  0.]
    end
    for k in 1:N
      if field[k] == 1.
        push!(states, "Up")
      elseif field[k] == -1.0
        push!(states, "Dn")
      else
        push!(states, "Z0")
      end
    end
    psi = productMPS(ComplexF64, sites, states)

  # elseif type == "stripes"
  #   spin_nb = 0
  #   for k in 1:N
  #     if k in spin_sites
  #       spin_nb += 1
  #       if spin_nb % 2 == 1
  #         push!(states, "Dn")
  #       else
  #         push!(states, "Up")
  #       end
  #     else
  #       push!(states, "Z0")
  #     end
  #   end
  #   psi = productMPS(ComplexF64, sites, states)

   elseif type == "stripes"
    line = 0 
    for k in 1:N
      if k % ly == 0
        line = 6
      else
        line = k % ly
      end
      if k in spin_sites
        if line % 2 == 1
          push!(states, "Up")
        elseif line % 2 == 0
          push!(states, "Dn")
        end
      else
        push!(states, "Z0")
      end
    end
    psi = productMPS(ComplexF64, sites, states)
  else
    psi = random_mps(sites;linkdims=2)
  end
  return psi
end




function ground_state_triangular(Lx, Ly, J, lambd, theta, D, ext_field, sites)
    N = Lx*Ly
    lattice = triangular_lattice(Lx,Ly; yperiodic = true)
    os1 = OpSum()
    os2 = OpSum()
    os3 = OpSum()
    # os4 = OpSum()
    # os5 = OpSum()

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

      ##uniform field, for FQ phase.
      #os5 += ext_field, "Sz", b.s1
    end

    ## symmetry breaking by pinning field: 
  #  sites_with_spins = spin_sites_afm(Lx, Ly)
  #  n_spins = length(sites_with_spins)
  #  local_field = pinning_field(n_spins, true)
  #  for l in 1:n_spins
  #     spin_site = sites_with_spins[l]
  #     constant = ext_field*local_field[l]
  #     println("sites:", spin_site, "field:", constant, "\n")
  #     os4 += constant, "Sz", spin_site
  #   end


    H1 = MPO(os1, sites)
    H2 = MPO(os2, sites)
    HD = MPO(os3, sites)
    # Hs = MPO(os4, sites)
    # Hmag = MPO(os5, sites)

    nsweeps = 10 #8  #10
    maxdim = 300 #300 #500
    cutoff = [1E-12] 

    #psi0 = random_mps(sites;linkdims=2)
    #psi0 = initialize_mps(lx, ly, sites, "stripes")
    #psi0 = initialize_mps(lx, ly, sites, "afm")
    psi0 = initialize_mps(lx, ly, sites, "wanier")
    energy,psi = dmrg([H1,H2,HD],psi0;nsweeps,maxdim,cutoff)
    return energy, psi
end


###generate resluts
#lx, ly = 10, 6
# N = lx*ly
# j = 1.
# lambd = 0.1 
# theta =  3*pi/8
# D = -0.2 
# ext_field = 0.

# sites = siteinds("S=1", lx*ly; conserve_qns = false)



###RANDOM WANIER STATE
# psi0 = initialize_mps(lx, ly, sites, "wanier")
# fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/psi0_wanier.h5", "w")
# write(fo, "MPS", psi0)
# close(fo)

# C_s = spin_correlations(psi0)
# C_q = quadrupolar_correlations(psi0, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/psi0_spin_wanier.txt", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/psi0_quadrupolar_wanier.txt", C_q)

# ###compute ground state energy
# e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
# println("field:", ext_field)
# println("ground state:", e/(lx*ly), "\n")
# ### download mps
# fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field)_ini_wanier.h5", "w")
# write(fo, "MPS", ψ)
# close(fo)
# ##download correl
# C_s = spin_correlations(ψ)
# C_q = quadrupolar_correlations(ψ, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/spin_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field)_ini_wanier.txt", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/quadrupolar_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field)_ini_wanier.txt", C_q)




lx, ly = 6, 6
N = lx*ly
j = 1.
lambd = 0.3
theta =  3*pi/8
D = -0.2 
ext_field = 0.
sites = siteinds("S=1", lx*ly; conserve_qns = false)


#WANNIER
e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/DMRG_wannier_lambd_$(lambd).h5", "w")
write(fo, "MPS", ψ)
close(fo)

C_s = spin_correlations(ψ)
C_q = quadrupolar_correlations( ψ, sites)
writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/spin_wannier_lambda_$(lambd).txt", C_s)
writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/quadrupolar_wannier_lambda_$(lambd).txt", C_q)






###STRIPES
# println("STRIPES")
# psi0 = initialize_mps(lx, ly, sites, "stripes")
# fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/stripes_ini.h5", "w")
# write(fo, "MPS", psi0)
# close(fo)
# C_s = spin_correlations(psi0)
# C_q = quadrupolar_correlations( psi0, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/spin_stripes_ini.txt", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/quadrupolar_stripes_ini.txt", C_q)




# e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
# fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/DMRG_stripes_lambd_$(lambd).h5", "w")
# write(fo, "MPS", ψ)
# close(fo)

# C_s = spin_correlations(ψ)
# C_q = quadrupolar_correlations( ψ, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/spin_stripes_lambda_$(lambd).txt", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/quadrupolar_stripes_lambda_$(lambd).txt", C_q)



###AFM 
# println("AFM")
# e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
# fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/DMRG_afm_lambd_$(lambd).h5", "w")
# write(fo, "MPS", ψ)
# close(fo)

# C_s = spin_correlations(ψ)
# C_q = quadrupolar_correlations( ψ, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/spin_afm_lambda_$(lambd).txt", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/quadrupolar_afm_lambda_$(lambd).txt", C_q)







###LAMBDA VARIABLE
# lx, ly = 16, 6 #6, 6 #10, 6
# N = lx*ly
# j = 1.
# theta =  3*pi/8
# D = -0.2 
# ext_field = 0.
# sites = siteinds("S=1", lx*ly; conserve_qns = false)



# for lambd in [0.05]
#   println("lambda = ", lambd, "\n")
#   psi0 = initialize_mps(lx, ly, sites, "afm")
  
#   e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
 
#   fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/DMRG_afm_lambd_$(lambd).h5", "w")
#   write(fo, "MPS", ψ)
#   close(fo)

#   C_s = spin_correlations(ψ)
#   C_q = quadrupolar_correlations( ψ, sites)
#   writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/spin_afm_lambda_$(lambd).txt", C_s)
#   writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/quadrupolar_afm_lambda_$(lambd).txt", C_q)
# end








# ###compute ground state energy
# e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
# println("field:", ext_field)
# println("ground state:", e/(lx*ly), "\n")
# ### download mps
# fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field)_ini_afm.h5", "w")
# write(fo, "MPS", ψ)
# close(fo)
# ##download correl
# C_s = spin_correlations(ψ)
# C_q = quadrupolar_correlations(ψ, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/spin_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field)_ini_afm.txt", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/quadrupolar_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field)_ini_afm.txt", C_q)


















# C_s = spin_correlations(psi0)
# C_q = quadrupolar_correlations(psi0, sites)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/psi0_spin_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(0)", C_s)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/psi0_quadrupolar_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(0)", C_q)


## QQ - AF test  
# for ext_field in  [0.0] 
#   e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
#   println("field:", ext_field)
#   println("ground state:", e/(lx*ly), "\n")
#   ### download mps
#   fo = h5open("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field)_ini_afm.h5", "w")
#   write(fo, "MPS", ψ)
#   close(fo)
#   ##download correl
#   C_s = spin_correlations(ψ)
#   C_q = quadrupolar_correlations(ψ, sites)
#   writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/spin_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field)_ini_afm.txt", C_s)
#   writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/quadrupolar_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field)_ini_afm.txt", C_q)
# end   



### QQ-F by increasing theta
# for D in [-0.5, -1.0, -1.5, -2.0, -2.2]
#   println("D = ", D)
#   for ext_field in  [0.02] 
#     e, ψ = ground_state_triangular(lx, ly, j, lambd, theta, D, ext_field, sites)
#     println("field:", ext_field)
#     println("ground state:", e/(lx*ly), "\n")
#     ### download mps
#     fo = h5open("C:/Users/aroge/Desktop/KITP/QQ_F/DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(ext_field).h5", "w")
#     write(fo, "MPS", ψ)
#     close(fo)

#     ##download correl
#     C_s = spin_correlations(ψ)
#     C_q = quadrupolar_correlations(ψ, sites)
#     writedlm("C:/Users/aroge/Desktop/KITP/QQ_F/spin_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field).txt", C_s)
#     writedlm("C:/Users/aroge/Desktop/KITP/QQ_F/quadrupolar_correl_theta_$(theta)_lambd_$(lambd)_D_$(D)_field_$(ext_field).txt", C_q)
#   end   
# end

