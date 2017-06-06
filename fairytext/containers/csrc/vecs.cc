#include <cstdint>
    
void fill_float_vec(float* in_vec, int64_t in_vec_len, float* out_vec, int64_t out_vec_len, float fill_value){
    int64_t i;
    #pragma GCC ivdep
    for (i = 0; i < in_vec_len; i++){
        out_vec[i] = in_vec[i];
    }
    #pragma GCC ivdep
    for (i = in_vec_len; i < out_vec_len; i++){
        out_vec[i] = fill_value;
    }
}
    
void fill_int_vec(int32_t* in_vec, int64_t in_vec_len, int32_t* out_vec, int64_t out_vec_len, int32_t fill_value){
    int64_t i;
    #pragma GCC ivdep
    for (i = 0; i < in_vec_len; i++){
        out_vec[i] = in_vec[i];
    }
    #pragma GCC ivdep
    for (i = in_vec_len; i < out_vec_len; i++){
        out_vec[i] = fill_value;
    }
}
    
void fill_short_vec(int16_t* in_vec, int64_t in_vec_len, int16_t* out_vec, int64_t out_vec_len, int16_t fill_value){
    int64_t i;
    #pragma GCC ivdep
    for (i = 0; i < in_vec_len; i++){
        out_vec[i] = in_vec[i];
    }
    #pragma GCC ivdep
    for (i = in_vec_len; i < out_vec_len; i++){
        out_vec[i] = fill_value;
    }
}
