#include <erl_nif.h>

static ERL_NIF_TERM ok(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    return enif_make_atom(env, "ok");
}

static ErlNifFunc nif_funcs [] =
{
    {"ok", 0, ok}
};

ERL_NIF_INIT(Elixir.NxSgemm, nif_funcs, NULL, NULL, NULL, NULL)
