import igl
import sympy as sp
import numpy as np
import argparse
sp.init_printing(use_unicode=True)

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="bunnyLowRes.obj")
parser.add_argument("--density", type=float, default=1000, help="density in kg/m^3")
parser.add_argument("--scale", nargs=3, type=float, default=(1,1,1),help="scale the mesh")
parser.add_argument("--translate", nargs=3, type=float, default=(0,0,0),help="translate (after scale)")
parser.add_argument("--test", type=int, help="run a numbered unit test")
args = parser.parse_args()

V, _, _, F, _, _ = igl.read_obj(args.file)

V = V * args.scale
V = V + args.translate


A = sp.Matrix(sp.symbols('a0 a1 a2'))
B = sp.Matrix(sp.symbols('b0 b1 b2'))
C = sp.Matrix(sp.symbols('c0 c1 c2'))
D = sp.Matrix([0,0,0])

alpha = sp.Symbol("alpha")
beta = sp.Symbol("beta")
gamma = sp.Symbol("gamma")
sub_x = alpha*A + beta*B + gamma*C + D-(alpha*D + beta*D + gamma*D)

volume_exp = 1/6*(A.dot(B.cross(C)))
volume_func = sp.lambdify((*A, *B, *C), volume_exp, 'numpy')
#volume = volume_func(*V[face[0]],*V[face[1]],*V[face[2]])

f_exp = sub_x*args.density
integrated_f = sp.integrate(f_exp,(gamma,0,1-alpha-beta))
integrated_f = sp.integrate(integrated_f,(beta,0,1-alpha))
integrated_f = sp.integrate(integrated_f,(alpha,0,1))

wc_func = sp.lambdify((*A, *B, *C), integrated_f,'numpy')


r = sub_x

rHat = sp.Matrix([[0,-1*r[2],r[1]],[r[2],0,-1*r[0]],[-1*r[1],r[0],0]])
rHatT = rHat.transpose()

integrated_f = sp.integrate(rHatT * rHat,(gamma,0,1-alpha-beta))
integrated_f = sp.integrate(integrated_f,(beta,0,1-alpha))
integrated_f = sp.integrate(integrated_f,(alpha,0,1))
j_func = sp.lambdify((*A,*B,*C), integrated_f)

rIntegral = sp.Matrix.integrate(rHat,(gamma,0,1-alpha-beta))
rIntegral = sp.Matrix.integrate(rIntegral,(beta,0,1-alpha))
rIntegral = sp.Matrix.integrate(rIntegral,(alpha,0,1))
rHatFunc = sp.lambdify((*A,*B,*C),rIntegral)



def fullMesh(FACES):
    vol = 0
    wc = np.zeros((1,3))
    J = np.zeros((3,3))
    rHatC = np.zeros((3,3))
    for face in FACES:
        volMesh = volume_func(*V[face[0]],*V[face[1]],*V[face[2]])
        vol += volMesh
        wc = np.add(wc,6*volMesh*np.transpose(wc_func(*V[face[0]],*V[face[1]],*V[face[2]])))
        J = np.add(J,6*volMesh*args.density*j_func(*V[face[0]],*V[face[1]],*V[face[2]]))
        rHatC = np.add(rHatC,6*volMesh*args.density*rHatFunc(*V[face[0]],*V[face[1]],*V[face[2]]))
    mass = vol*args.density
    return (round(vol,3),round(mass,3),wc/mass,np.round(J,3),np.round(rHatC,3))

vol = volume_func(*V[F[0][0]],*V[F[0][1]],*V[F[0][2]])
if args.test == 1:
    print(f"vol = {vol}\nmass = {args.density* vol}")

if args.test == 2:
    wc = 6*vol*np.transpose(wc_func(*V[F[0][0]],*V[F[0][1]],*V[F[0][2]]))
    print(f"weighted com = {wc}")
    print(wc.type())

if args.test == 3:
    print(f"J = \n{6*vol*args.density*j_func(*V[F[0][0]],*V[F[0][1]],*V[F[0][2]])}")

vals = fullMesh(F)
print(f"volume = {vals[0]}\nmass = {vals[1]}")
print(f"com = {vals[2]}")
print(f"J = \n{vals[3]}")
massMatrix = np.zeros((6,6))
massMatrix[:3, :3] = np.eye(3)*vals[1]
massMatrix[:3, 3:] = vals[4].transpose()
massMatrix[3:, :3] = vals[4]
massMatrix[3:, 3:] = vals[3]
print(f"M =\n{massMatrix}")
