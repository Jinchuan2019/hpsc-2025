#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N],a[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    a[i] = i;
  }
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 avec = _mm512_load_ps(a);
  __m512 zerovec = _mm512_setzero_ps();
  for(int i=0; i<N; i++) {
    __m512 xivec =  _mm512_set1_ps(x[i]);
    __m512 yivec =  _mm512_set1_ps(y[i]);
    __m512 ivec =  _mm512_set1_ps(i);
    __mmask16 mask = _mm512_cmp_ps_mask(ivec, avec, _MM_CMPINT_NE );
    __m512 rxvec = _mm512_sub_ps(xivec,xvec);
    __m512 ryvec = _mm512_sub_ps(yivec,yvec);
    __m512 rxsvec = _mm512_mul_ps(rxvec,rxvec);
    __m512 rysvec = _mm512_mul_ps(ryvec,ryvec);
    __m512 rssvec = _mm512_add_ps(rxsvec,rysvec);
    __m512 rrvec = _mm512_rsqrt14_ps(rssvec);
    __m512 fxivec = _mm512_mul_ps(rxvec,mvec);
    fxivec = _mm512_mul_ps(fxivec,rrvec);
    fxivec = _mm512_mul_ps(fxivec,rrvec);
    fxivec = _mm512_mul_ps(fxivec,rrvec);
    fxivec = _mm512_mask_blend_ps(mask, zerovec, fxivec);
    fx[i] -= _mm512_reduce_add_ps(fxivec);
    __m512 fyivec = _mm512_mul_ps(ryvec,mvec);
    fyivec = _mm512_mul_ps(fyivec,rrvec);
    fyivec = _mm512_mul_ps(fyivec,rrvec);
    fyivec = _mm512_mul_ps(fyivec,rrvec);
    fyivec = _mm512_mask_blend_ps(mask, zerovec, fyivec);
    fy[i] -= _mm512_reduce_add_ps(fyivec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }
