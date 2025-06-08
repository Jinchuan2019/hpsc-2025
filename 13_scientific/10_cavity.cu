#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;
typedef vector<vector<float>> matrix;

__global__ void ComputeB(float *u, float *v,float *b,double rho,double nu,int nx,double dx,double dy,double dt) {
  int j = threadIdx.x+1;
  for (int i=1; i<nx-1; i++) {
    // Compute b[j][i]
    float dudx = (u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx);
    float dvdy = (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy);
    b[j*nx+i] = rho * (1 / dt *
              (dudx + dvdy) -
              dudx*dudx - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *
              (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx)) - dvdy*dvdy);
  }
}
__global__ void ComputeP(float *p, float *pn,float *b,int nx,double dx,double dy,double dt) {
  int j = threadIdx.x+1;
  for (int i=1; i<nx-1; i++) {
  // Compute p[j][i]
    p[j*nx+i] = (dy * dy * (pn[j*nx+i+1] + pn[j*nx+i-1]) +
      dx * dx * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -
      b[j*nx+i] * dx * dx * dy * dy)
    / (2 * (dx*dx + dy*dy));
  }
}
__global__ void ComputeUV(float *u, float *un,float *v, float *vn,float *p,double rho,double nu,int nx,double dx,double dy,double dt) {
  int j = threadIdx.x+1;
  for (int i=1; i<nx-1; i++) {
  // Compute u[j][i] and v[j][i]
  u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i - 1])
                          - vn[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i])
                          - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                          + nu * dt / (dx*dx) * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1])
                          + nu * dt / (dy*dy) * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
  v[j*nx+i] = vn[j*nx+i] - un[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i - 1])
                          - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i])
                          - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                          + nu * dt / (dx*dx) * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1])
                          + nu * dt / (dy*dy) * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);

  }
}
int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dx2=dx*dx;
  double dy2=dy*dy;
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  float *u;cudaMallocManaged(&u,ny*nx*sizeof(float));
  float *v;cudaMallocManaged(&v,ny*nx*sizeof(float));
  float *p;cudaMallocManaged(&p,ny*nx*sizeof(float));
  float *b;cudaMallocManaged(&b,ny*nx*sizeof(float));
  float *un;cudaMallocManaged(&un,ny*nx*sizeof(float));
  float *vn;cudaMallocManaged(&vn,ny*nx*sizeof(float));
  float *pn;cudaMallocManaged(&pn,ny*nx*sizeof(float));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j*nx+i] = 0;
      v[j*nx+i] = 0;
      p[j*nx+i] = 0;
      b[j*nx+i] = 0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
    ComputeB<<<1,ny-2>>>(u,v,b,rho,nu,nx,dx,dy,dt);
    cudaDeviceSynchronize();
    for (int it=0; it<nit; it++) {
      for (int ij=0; ij<ny*nx; ij++)
	      pn[ij] = p[ij];
      ComputeP<<<1,ny-2>>>(p,pn,b,nx,dx,dy,dt);
      cudaDeviceSynchronize();
      for (int j=0; j<ny; j++) {
        // Compute p[j][0] and p[j][nx-1]
        p[j*nx+0]=p[j*nx+1];
        p[j*nx+nx-1]=p[j*nx+nx-2];
      }
      for (int i=0; i<nx; i++) {
	      // Compute p[0][i] and p[ny-1][i]
        p[0*nx+i]= p[1*nx+i];
        p[(ny-1)*nx+i]= 0;
      }
    }
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j*nx+i] = u[j*nx+i];
	      vn[j*nx+i] = v[j*nx+i];
      }
    }
    ComputeUV<<<1,ny-2>>>(u,un,v,vn,p,rho,nu,nx,dx,dy,dt);
    cudaDeviceSynchronize();
    for (int j=0; j<ny; j++) {
      // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      u[j*nx+0]=0;
      u[j*nx+nx-1]=0;
      v[j*nx+0]=0;
      v[j*nx+nx-1]=0;
    }
    for (int i=0; i<nx; i++) {
      // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      u[0*nx+i]=0;
      u[(ny-1)*nx+i]=1;
      v[0*nx+i]=0;
      v[(ny-1)*nx+i]=0;
    }
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j*nx+i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j*nx+i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j*nx+i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
}
