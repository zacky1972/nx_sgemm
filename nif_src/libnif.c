#include <erl_nif.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef USE_OPEN_BLAS
#include <cblas.h>
#else // USE_OPEN_BLAS
#include <Accelerate/Accelerate.h>
#endif // USE_OPEN_BLAS

static ERL_NIF_TERM ok(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM mul_nif_f32_tensor_f32_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    if (__builtin_expect(argc != 3, false)) {
        return enif_make_badarg(env);
    }

    ErlNifUInt64 vec_size;
    if (__builtin_expect(!enif_get_uint64(env, argv[0], &vec_size), false)) {
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM binary_term = argv[1];
    ErlNifBinary in_data;
    if (__builtin_expect(!enif_inspect_binary(env, binary_term, &in_data), false)) {
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM double_term = argv[2];
    double factor;
    if (__builtin_expect(!enif_get_double(env, double_term, &factor), false)) {
        return enif_make_badarg(env);
    }

    float *in = (float *)in_data.data;
    ErlNifBinary out_data;
    if (__builtin_expect(!enif_alloc_binary(vec_size * sizeof(float), &out_data), false)) {
        return enif_make_badarg(env);
    }

    float *out = (float *)out_data.data;

    cblas_scopy((int)vec_size, in, 1, out, 1);
    cblas_sscal((int)vec_size, (float) factor, out, 1);

    return enif_make_binary(env, &out_data);
}

static ERL_NIF_TERM mul_nif_u8_tensor_u8_scalar(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    if (__builtin_expect(argc != 3, false)) {
        return enif_make_badarg(env);
    }

    ErlNifUInt64 vec_size;
    if (__builtin_expect(!enif_get_uint64(env, argv[0], &vec_size), false)) {
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM binary_term = argv[1];
    ErlNifBinary in_data;
    if (__builtin_expect(!enif_inspect_binary(env, binary_term, &in_data), false)) {
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM uint_term = argv[2];
    unsigned int factor;
    if (__builtin_expect(!enif_get_uint(env, uint_term, &factor), false)) {
        return enif_make_badarg(env);
    }

    uint8_t *in = (uint8_t *)in_data.data;
    ErlNifBinary out_data;
    if (__builtin_expect(!enif_alloc_binary(vec_size * sizeof(uint8_t), &out_data), false)) {
        return enif_make_badarg(env);
    }

    uint8_t *out = (uint8_t *)out_data.data;

    for(ErlNifUInt64 i = 0; i < vec_size; i++) {
        out[i] = (uint8_t) (in[i] * factor); 
    }

    return enif_make_binary(env, &out_data);
}

static ERL_NIF_TERM dot_nif_f32_matrix_f32_matrix(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    if (__builtin_expect(argc != 5, false)) {
        return enif_make_badarg(env);
    }

    ErlNifUInt64 m;
    if (__builtin_expect(!enif_get_uint64(env, argv[0], &m), false)) {
        return enif_make_badarg(env);
    }

    ErlNifUInt64 o;
    if (__builtin_expect(!enif_get_uint64(env, argv[1], &o), false)) {
        return enif_make_badarg(env);
    }

    ErlNifUInt64 n;
    if (__builtin_expect(!enif_get_uint64(env, argv[2], &n), false)) {
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM binary_term_a = argv[3];
    ErlNifBinary a_data;
    if (__builtin_expect(!enif_inspect_binary(env, binary_term_a, &a_data), false)) {
        return enif_make_badarg(env);
    }
    float *a = (float *)a_data.data;

    ERL_NIF_TERM binary_term_b = argv[4];
    ErlNifBinary b_data;
    if (__builtin_expect(!enif_inspect_binary(env, binary_term_b, &b_data), false)) {
        return enif_make_badarg(env);
    }
    float *b = (float *)b_data.data;

    ErlNifBinary c_data;
    if (__builtin_expect(!enif_alloc_binary(m * o * sizeof(float), &c_data), false)) {
        return enif_make_badarg(env);
    }
    float *c = (float *)c_data.data;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, o, n, 1.0, a, n, b, o, 0.0, c, o);

    return enif_make_binary(env, &c_data);
}

static ErlNifFunc nif_funcs [] =
{
    {"ok", 0, ok},
    {"mul_nif_f32_tensor_f32_scalar", 3, mul_nif_f32_tensor_f32_scalar},
    {"mul_nif_u8_tensor_u8_scalar", 3, mul_nif_u8_tensor_u8_scalar},
    {"dot_nif_f32_matrix_f32_matrix", 5, dot_nif_f32_matrix_f32_matrix}
};

ERL_NIF_INIT(Elixir.NxSgemm, nif_funcs, NULL, NULL, NULL, NULL)
