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

  @doc """
  Element-wise multiplication of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do not match and broadcasting is possible.

  ## Examples

  ### Multiplying scalers

      iex> NxSgemm.multiply(1, 2)
      #Nx.Tensor<
        s32
        2
      >

  ### Multiplying tensors and scalers

      iex> NxSgemm.multiply(Nx.tensor([1, 2, 3], names: [:data], type: :u8), 1)
      #Nx.Tensor<
        u8[data: 3]
        [1, 2, 3]
      >

      iex> NxSgemm.multiply(1, Nx.tensor([1, 2, 3], names: [:data], type: :u8))
      #Nx.Tensor<
        u8[data: 3]
        [1, 2, 3]
      >
  """
  def multiply(a, b) when is_integer(a) and is_integer(b) do
    Nx.tensor(a * b, type: :s32)
  end

  def multiply(a, b) when is_integer(b) when 0 <= b and b < 256 do
    case Nx.type(a) do
      {:u, 8} ->
        %{
          a
          | data: %{
            a.data
            | state: mul_nif_u8_tensor_u8_scalar(Nx.size(a), a.data.state, b)
          }
        }
    end
  end

  def multiply(a, b) when is_integer(a) when 0 <= a and a < 256 do
    multiply(b, a)
  end

  defp mul_nif_u8_tensor_u8_scalar(_size, _a, _b), do: raise("NIF mul_nif_u8_tensor_u8_scalar/3 not implemented")
end
