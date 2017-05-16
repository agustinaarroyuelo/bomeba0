"""
Use Values taken from GAFF force field for atom types c, hc, o, n, s
and precompute values for par_s_ij and par_eps_ij.

"""
par_lj = {'C':(1.908, 0.086), 'H':(1.487, 0.0157), 
          'O':(1.6612, 0.21), 'N':(1.824, 0.17), 
          'S':(2., 0.25)}

par_s_ij = {}
par_eps_ij = {}
for i in 'CHONS':
    for j in 'CHONS':
        par_s_ij[i+j] = par_lj[i][0] + par_lj[j][0]
        par_eps_ij[i+j] = round((par_lj[i][1] * par_lj[j][1])**0.5, 5)
print(par_s_ij)
print(par_eps_ij)
