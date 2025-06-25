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
    field = [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  0.,
       -1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
        1.,  0.,  0., -1.,  0., -1.,  0.,  0., -1.,  0., -1.,  0.,  0.,
       -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0., -1.,  0.,  0.,
       -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]

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

  elseif type == "stripes"
    spin_nb = 0
    for k in 1:N
      if k in spin_sites
        spin_nb += 1
        if spin_nb % 2 == 1
          push!(states, "Dn")
        else
          push!(states, "Up")
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


###define quadrupolar operators
function ITensors.op(::OpName"Qxx", ::SiteType"S=1", s::Index)
    I = op("Id", s)
    Sx2 = op("Sx2", s)
    return Sx2 - 2/3*I
end

function ITensors.op(::OpName"Qyy", ::SiteType"S=1", s::Index)
    I = op("Id", s)
    Sy2 = op("Sy2", s)
    return Sy2 - 2/3*I
end

function ITensors.op(::OpName"Qzz", ::SiteType"S=1", s::Index)
    I = op("Id", s)
    Sz2 = op("Sz2", s)
    return Sz2 - 2/3*I
end

function ITensors.op(::OpName"Qxy", ::SiteType"S=1", s::Index)
    Sxy = op("Sx * Sy", s)
    Syx = op("Sy * Sx", s)
    return (Sxy + Syx)/2
end

function ITensors.op(::OpName"Qxz", ::SiteType"S=1", s::Index)
    Sxz = op("Sx * Sz", s)
    Szx = op("Sz * Sx", s)
    return (Sxz + Szx)/2
end

function ITensors.op(::OpName"Qyz", ::SiteType"S=1", s::Index)
    Syz = op("Sy * Sz", s)
    Szy = op("Sz * Sy", s)
    return (Syz + Szy)/2
end



function ITensors.op(::OpName"Qyx", ::SiteType"S=1", s::Index)
    Sxy = op("Sy * Sx", s)
    Syx = op("Sx * Sy", s)
    return (Sxy + Syx)/2
end

function ITensors.op(::OpName"Qzx", ::SiteType"S=1", s::Index)
    Sxz = op("Sz * Sx", s)
    Szx = op("Sx * Sz", s)
    return (Sxz + Szx)/2
end

function ITensors.op(::OpName"Qzy", ::SiteType"S=1", s::Index)
    Syz = op("Sz * Sy", s)
    Szy = op("Sy * Sz", s)
    return (Syz + Szy)/2
end



##QUANTUM FLUCTUATIONS. Pure classical: = 4/3
function Q_fluctuations(psi, Q_operator)
    mean_Q = expect(psi, Q_operator)

    # mean_Q = sum(mean_Q)/length(mean_Q)
    # return mean_Q^2

    mean_Q2 = mean_Q.^2
    return sum(mean_Q2)/length(mean_Q2)
end

function S_fluctuations(psi, S_operator)
    mean_S = expect(psi, S_operator)

    # mean_S = sum(mean_S)/length(mean_S)
    # return mean_S^2

    mean_S2 = mean_S.^2
    return sum(mean_S2)/length(mean_S2)

end

function quantum_fluctuations(psi)
    res = 0.
    for Q_operator in ["Qxx", "Qyy", "Qzz", "Qxy", "Qyx", "Qxz", "Qzx", "Qyz", "Qzy"]
        res += Q_fluctuations(psi, Q_operator)
    end
    res *= 0.5
    for S_operator in ["Sx", "Sy", "Sz"]
        res += S_fluctuations(psi, S_operator)
    end
    return res
end



###SUPERSOLID STATES: <mx^2 + my^2> vs <mz>? 
function orthogonal_magnetization(psi)
    #mean_m_ortho2 = expect(psi, "Sx * Sx + Sy * Sy")
    mean_mx = expect(psi, "Sx * Sx") #.^2
    mean_my = expect(psi, "Sy * Sy") #.^2
    mean_m_ortho2 = mean_mx + mean_my

    return sqrt(sum(mean_m_ortho2))/length(mean_m_ortho2)
end


function magnetization_z2(psi)
    mean_m = expect(psi, "Sz * Sz")
    return sum(mean_m)/length(mean_m)
end

###using definition from balenzs etc : https://journals.aps.org/prb/pdf/10.1103/PhysRevB.79.020409


function orthogonal_correls(psi)
    corr = correlation_matrix(psi,"S+","S-")
    return corr
end

function z_correls(psi)
    corr = correlation_matrix(psi, "Sz", "Sz")
    return corr 
end



###<mag_ortho^2>

function mag_ortho_2(psi)
    mean_mx = expect(psi, "Sp * Sm * Sp * Sm")
    mean_m_ortho2 = mean_mx
    return sqrt(sum(mean_m_ortho2))/length(mean_m_ortho2)

end


function correl_spsm2(psi)
  corr = real(correlation_matrix(psi,"S+ * S-","S+ * S-"))
  return corr
end


# ###xxz spin half triangular 
# j2 = 0.02
# j1 = 1.
# path = "C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/"
# filename = "psi_delta_$(j2/j1).h5"
# psi = load_mps_v2(filename, path)
# println("mag2 = ", magnetization_z2(psi))


###XXZ triangulaire triangular_XXZ_half

d = 0.02
path = "C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/"
filename = "psi_delta_$(d).h5"
psi = load_mps_v2(filename, path)

correl = correl_spsm2(psi)
writedlm("C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/ortho2_delta_$(d)_afm.txt", correl)

# println("mag2 = ", magnetization_z2(psi), "\n")
# println("<m_ortho^2> = ", mag_ortho_2(psi))







##recreate correlator for magnetization
# d = 0.3
# path = "C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/16_6/"
# #path = "C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/lambd_0.4/"
# filename = "DMRG_afm_lambd_$(d).h5"
# psi = load_mps_v2(filename, path)

# correl = correl_spsm2(psi)
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/16_6/ortho2_delta_$(d)_afm.txt", correl)

# println("mag2 = ", magnetization_z2(psi), "\n")
# println("<m_ortho^2> = ", mag_ortho_2(psi))

# Szcorr = z_correls(psi)
# Soth = orthogonal_correls(psi)

# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/fss_wannier/16_6/ortho_correl_delta_$(d)_wannier.txt", real(Soth))
# writedlm("C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/fss_wannier/16_6/Sz_correl_delta_$(d)_wannier.txt", real(Szcorr))





## test sur ferromagnet manually implemented. return 1.083 for quantum fluctuations.
# N = 200
# sites = siteinds("S=1", N)
# states = ["Up" for n=1:N]
# psi = productsMPS(ComplexF64, sites, states)
# println("Ferro (artificial) quantum fluctuations:", quantum_fluctuations(psi), "\n")
# println("ortho magnetization:", orthogonal_magnetization(psi), "\n")
# println("Sz magnetization:", magnetization_z(psi))



####XXZ triangular
# for delta in [0.05, 1., 2., 3., 4., 5., 6., 8.]
#     path = "C:/Users/aroge/Desktop/KITP/triangular_XXZ_half/results/"
#     filename = "psi_delta_$(delta).h5"
#     psi = load_mps_v2(filename, path)
#     println("ortho magnetization:", orthogonal_magnetization(psi), "\n")
#     println("Sz magnetization:", magnetization_z(psi), "\n")
# end






# ###test sur AFM final state 
# J = 1.0
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.0
# lx, ly = 10, 6


# # ##afm
# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "psi0_afm.h5"
# psi2 = load_mps_v2(filename, path)

# println("AFM INI", "\n")
# println("quantum fluctuations:", quantum_fluctuations(psi2), "\n")


# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(field)_ini_afm.h5"
# psi = load_mps_v2(filename, path)

# println("AFM FINAL", "\n")
# println("quantum fluctuations:", quantum_fluctuations(psi), "\n")

# ##wanier
# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "psi0_wanier.h5"
# psi0 = load_mps_v2(filename, path)

# println("WANIER INI", "\n")
# println("quantum fluctuations:", quantum_fluctuations(psi0), "\n")


# path = "C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/"
# filename = "DMRG_D_$(D)_lambd_$(lambd)_theta_$(theta)_field_$(field)_ini_wanier.h5"
# psi = load_mps_v2(filename, path)

# println("WANIER FINAL", "\n")
# println("quantum fluctuations:", quantum_fluctuations(psi), "\n")

# println("ortho magnetization:", orthogonal_magnetization(psi), "\n")
# println("Sz magnetization:", magnetization_z(psi))




# ###test sur QQ-AF
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.12

# psi = load_mps("C:/Users/aroge/Desktop/KITP/FSS_field/edge_field/", D, lambd, theta, field)
# println("QQ-AF quantum fluctuations:", quantum_fluctuations(psi), "\n")



# ###test sur QQ- fero phase.
# theta = 2.5*pi/4
# D = - 2.0
# lambd = 0.1
# field = 0.02
# psi = load_mps("C:/Users/aroge/Desktop/KITP/QQ_F", D, lambd, theta, field)

# println("QQ-F quantum fluctuations:", quantum_fluctuations(psi), "\n")

# # QQ AF ordonné
# theta = 1.1780972450961724
# D = -0.2
# lambd = 0.1
# field = 0.075

# psi = load_mps("C:/Users/aroge/Desktop/KITP/FSS_field/lx_10_ly_6/", D, lambd, theta, field)
# println("QQ-AF quantum fluctuations:", quantum_fluctuations(psi), "\n")


