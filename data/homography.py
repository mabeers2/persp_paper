import numpy as np 
from utils import to_homogeneous, SimilarityTransform

n_pts = 10
right = np.random.uniform(0,10, size=(n_pts,3))
left = right @ np.diag([-1,1,1.])
pts = np.zeros((2 * n_pts, 3))
pts[::2] = right
pts[1::2] = left

st = SimilarityTransform(123,75,231,0,0,15,1)
xyz = st.apply(pts)
ortho = to_homogeneous(xyz[:, :2])
persp = to_homogeneous(xyz[:, :2] / xyz[:, 2][:, None])
H = np.linalg.inv(persp.T @ persp) @ persp.T @ ortho
np.linalg.norm(persp @ H - ortho)


C = xyz[::2] - xyz[1::2]
C /= np.linalg.norm(C, axis=1, keepdims=True)
n = C[0]
u,s,vt = np.linalg.svd(n[:, None])
B = u[:, 1:]
# n /= np.linalg.norm(n)
I = np.eye(3)
p = persp[0]
q = persp[1]
UR = np.vstack((n, -p, np.zeros(3))).T
LR = np.vstack((-n, np.zeros(3), -q)).T
c = np.hstack([n, 0,0,0])
B = np.block([[I, UR], [I, LR]])
A = np.block([[B, c[:, None]], [c, 0]])
d = 1
b = np.hstack((np.zeros(6),-d))
A_inv = np.linalg.pinv(A)


M = np.eye(7) - A_inv @ A
x = A_inv @ b + M @ (100*np.random.randn(7))
midpoint = x[:3]
D, s,t, lam = x[3:]

print(np.isclose(n @ midpoint + d, 0))

p1 = midpoint + D * n
q1 = midpoint - D * n


mx,my,mz = midpoint
nx,ny,mz = n



mx + D *nx - s * p[0]



d = 2
C = xyz[::2] - xyz[1::2]
C /= np.linalg.norm(C, axis=1, keepdims=True)
n = C[0]
u,s,vt = np.linalg.svd(n[:, None])
B = u[:, 1:]
A = np.block([
	[B, n[:, None], -p[:, None], np.zeros((3,1))],
	[B, -n[:, None], np.zeros((3,1)), -q[:, None]]
])
b = np.hstack((d*n, d*n))
params = np.linalg.solve(A.T @ A, A.T@b)
midpoint = B @ params[:2] - d * n
D,s,t = params[2:]
P_hat2 = midpoint + D * n
Q_hat2 = midpoint - D * n



# target = np.linalg.inv(A.T @ A)
# A_s = 2 * np.eye(3) 
# B_s = -np.vstack((
# 	np.hstack([B.T @ p[:, None], B.T @ q[:, None]]),
# 	np.array([n@p, -n@q])
# ))
# C_s = B_s.T
# D_s = np.diag([(p@p), (q@q)])
# A_s_inv = np.eye(3) / 2
# A2 = np.block([[A_s,B_s],[C_s,D_s]])



# S_inv = np.diag([1/2,1/2,1/2,1 / (p@p - .5), 1 / (q@q - .5)])
# M1 = np.block([[np.eye(3), -A_s_inv @ B_s],[np.zeros((2,3)), np.eye(2)]])
# S = D_s - .5 * C_s @ B_s
# M2 = np.block([[np.eye(3)/2, np.zeros((3,2))],[np.zeros((2,3)), np.linalg.inv(S)]])
# ata_inv = M1.T @ M2 @ M1

# D_inv = np.linalg.inv(S)
# C_inv = -D_inv @ C_s @ A_s_inv
# B_inv = -A_s_inv @ B_s @ D_inv
# A_inv = A_s_inv + A_s_inv @ B_s @ D_inv @ C_s @ A_s_inv
# abcd_inv = np.block([[A_inv, B_inv],[C_inv, D_inv]])




def persp_to_3D(data: VanishingPointData, sym_plane_intercept: float=1.0):
	"""
	Unique Reconstruction of each symmetry correspondence independently. 
	Produces 3D output even with noisy image. 
	-----------------------------------------
	Concept: 
	(1) Suppose plane given by 
	"""
	vph = get_vanishing_point(data, homogeneous=True)






import sympy as sp
d,s,t, D = sp.symbols("d s t D")
B = sp.MatrixSymbol('B', 3, 2)
p = sp.MatrixSymbol('p', 3, 1)
q = sp.MatrixSymbol('q', 3, 1)
n = sp.MatrixSymbol('n', 3, 1)
alpha = sp.MatrixSymbol('alpha', 2, 1)
u = -d * n  + B * alpha + D * n - s * p
v = -d * n  + B * alpha - D * n - t * q
cost = u.T * u + v.T * v

eqs = sp.diff(cost, s)
eqt = sp.diff(cost, t)
eqD = sp.diff(cost, D)
eqalpha = sp.diff(cost, alpha)
context = sp.Q.orthogonal(B)
sp.solve([eqs, eqt, eqD], (s,t,D), dict=True)












