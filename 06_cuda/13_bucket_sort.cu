#include <cstdio>
#include <cstdlib>
#include <vector>
__global__ void init_bucket(int *a) {
  a[threadIdx.x] = 0;
 }
__global__ void bucket_count_self(int *key,int n,int* bucket) {
  int c = 0;
  for (int i=0; i<n; i++){
    if (key[i]<=threadIdx.x){
        c++;
    }
  }
  bucket[threadIdx.x]=c;
}
__global__ void recover_from_bucket(int *key,int n,int* bucket){
  int start = 0;
  int end = bucket[threadIdx.x];
  if(threadIdx.x>0){
    start=bucket[threadIdx.x-1];
  }
  for(int i=start;i<end;i++){
    key[i]=threadIdx.x;
  }
}
int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int* a_key;
  cudaMallocManaged(&a_key, n*sizeof(float));
  for (int i=0; i<n; i++) {
    a_key[i] = key[i];
  }

  int* bucket;
  cudaMallocManaged(&bucket, range*sizeof(float));
  init_bucket<<<1,range>>>(bucket);
  bucket_count_self<<<1,range>>>(a_key,n,bucket);
  recover_from_bucket<<<1,range>>>(a_key,n,bucket);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    key[i] = a_key[i];
  }
  cudaFree(a_key);
  cudaFree(bucket);
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}