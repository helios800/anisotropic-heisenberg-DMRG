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




function correl_S_Q(psi, sites)
    Sx_Q, Sy_Q, Sz_Q = [], [], []
    for quad_operator in ["Qxx * Qxx", "Qxy * Qxy", "Qxz * Qxz", "Qxy * Qxy", "Qyy * Qyy", "Qyz * Qyz", "Qxz * Qxz", "Qyz * Qyz", "Qzz * Qzz"] #["Qxx", "Qxy", "Qxz", "Qxy", "Qyy", "Qyz", "Qxz", "Qyz", "Qzz"]
        t1 = correlation_matrix(complex(psi), "Sx * Sx", quad_operator)
        t2 = correlation_matrix(complex(psi), "Sy * Sy", quad_operator)
        t3 = correlation_matrix(complex(psi), "Sz * Sz", quad_operator)

        push!(Sx_Q, t1)
        push!(Sy_Q, t2)
        push!(Sz_Q, t3) 
    end
    return Sx_Q .+  Sy_Q .+ Sz_Q
end



function diagonalise_correlator(psi, site_i, site_j, correlator_list)
    correl_site = []
    for k in 1:length(correlator_list)
        push!(correl_site, correlator_list[k][site_i, site_j])
    end
    matrix = real(reshape(correl_site, 3, 3))
    eigvals, eigvecs = eigen(matrix)
    return eigvals
end


function max_spectrum_sites(psi, sites, lx, ly)
    N = lx*ly
    res = zeros(N, N)
    correlator_list = correl_S_Q(psi, sites)
    for site_i in 1:N
        for site_j in 1:N
            res[site_i, site_j] = maximum(diagonalise_correlator(psi,site_i, site_j, correlator_list))
        end
    end
    return res
end


###test sans DMRG
# lx, ly = 6, 6
# sites = siteinds("S=1", lx*ly; conserve_qns = false)
# psi = initialize_mps(lx, ly, sites, "wannier")

# res = max_spectrum_sites(psi, sites, lx, ly)
# println(res)
# writedlm("C:/Users/aroge/Desktop/KITP/test_sq_correls_wannier.txt", res)


###test sur etat optimaux
lx, ly = 6, 6
path = "C:/Users/aroge/Desktop/KITP/FSS_field/lambd_variable/fss_wannier/6_6/"
filename = "DMRG_wannier_lambd_0.3.h5"
psi = load_mps_v2(filename, path)

sites = siteinds(psi)
res = max_spectrum_sites(psi, sites, lx, ly)
println(res)
writedlm("C:/Users/aroge/Desktop/KITP/sq_correls_wannier_lambd_0.3.txt", res)