

## BP1 parameters
sim_years = 3000.

Vp = 1e-9 # plate rate
ρ = 2.670
cs = 3.464
σn = 50
RSamin = 0.01
RSamax = 0.025
RSb = 0.015
RSDc = 0.008 # change it to be 0.008
RSf0 = 0.6
RSV0 = 1e-6
RSVinit = 1e-9
RSH1 = 15
RSH2 = 18

μshear = cs^2 * ρ
η = μshear / (2 * cs)


## Domain size
Lx = 80
Ly = 80

(kx,ky) = (π/Lx, π/Ly)

## SBP parameters
SBPp = 2
bc_map = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN,
BC_JUMP_INTERFACE]


(x1,x2,x3,x4) = (0,1,0,2)
(y1,y2,y3,y4) = (0,0,1,2)

ex = [(α) -> x1 * (1 .- α) / 2 + x3 * (1 .+ α) / 2,
(α) -> x2 * (1 .- α) / 2 + x4 * (1 .+ α) / 2,
(α) -> x1 * (1 .- α) / 2 + x2 * (1 .+ α) / 2,
(α) -> x3 * (1 .- α) / 2 + x4 * (1 .+ α) / 2]

exα = [(α) -> -x1 / 2 + x3 / 2,
 (α) -> -x2 / 2 + x4 / 2,
 (α) -> -x1 / 2 + x2 / 2,
 (α) -> -x3 / 2 + x4 / 2]

ey = [(α) -> y1 * (1 .- α) / 2 + y3 * (1 .+ α) / 2,
(α) -> y2 * (1 .- α) / 2 + y4 * (1 .+ α) / 2,
(α) -> y1 * (1 .- α) / 2 + y2 * (1 .+ α) / 2,
(α) -> y3 * (1 .- α) / 2 + y4 * (1 .+ α) / 2]

eyα = [(α) -> -y1 / 2 + y3 / 2,
 (α) -> -y2 / 2 + y4 / 2,
 (α) -> -y1 / 2 + y2 / 2,
 (α) -> -y3 / 2 + y4 / 2]

 
xt(r,s) = transfinite_blend(ex[1], ex[2], ex[3], ex[4],
 exα[1], exα[2], exα[3], exα[4],
 r, s)

yt(r,s) = transfinite_blend(ey[1], ey[2], ey[3], ey[4],
 eyα[1], eyα[2], eyα[3], eyα[4],
 r, s)