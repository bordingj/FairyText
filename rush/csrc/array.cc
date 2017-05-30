
#include <cstdint>

void ftake(float* a, int64_t* indices, float* out, int64_t minibatchsize, int64_t M){
    int64_t m;
    int64_t n;
    int64_t idx;
    float* out_n;
    float* a_n;
    for (n=0; n<minibatchsize; n++){
        idx = indices[n];
        out_n = &out[n*M];
        a_n = &a[idx*M];
        #pragma GCC ivdep 
        for (m=0; m<M; m++){
            out_n[m] = a_n[m];
        }
    }
}

void itake(int64_t* a, int64_t* indices, int64_t* out, int64_t minibatchsize, int64_t M){
    int64_t m;
    int64_t n;
    int64_t idx;
    int64_t* out_n;
    int64_t* a_n;
    for (n=0; n<minibatchsize; n++){
        idx = indices[n];
        out_n = &out[n*M];
        a_n = &a[idx*M];
        #pragma GCC ivdep 
        for (m=0; m<M; m++){
            out_n[m] = a_n[m];
        }
    }
}
        
void fcpy(float* a, float* out, int64_t M){
    #pragma GCC ivdep 
    for (int64_t m=0; m<M; m++){
        out[m] = a[m];
    }
}

void icpy(int64_t* a, int64_t* out, int64_t M){
    #pragma GCC ivdep 
    for (int64_t m=0; m<M; m++){
        out[m] = a[m];
    }
}