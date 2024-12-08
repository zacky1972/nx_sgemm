defmodule NxSgemm do
  @moduledoc """
  Documentation for `NxSgemm`.
  """
  require Logger

  @on_load :load_nif

  @doc false
  def load_nif do
    nif_file = ~c'#{Application.app_dir(:nx_sgemm, "priv/libnif")}'

    case :erlang.load_nif(nif_file, 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} -> Logger.error("Failed to load NIF: #{inspect(reason)}")
    end
  end

  @doc """
  ok.

  ## Examples

      iex> NxSgemm.ok()
      :ok

  """
  def ok(), do: :erlang.nif_error(:not_loaded)
end
