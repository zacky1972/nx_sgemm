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

      iex> NxSgemm.multiply().(1, 2)
      #Nx.Tensor<
        s32
        2
      >

  ### Multiplying tensors and scalers

      iex> NxSgemm.multiply().(Nx.tensor([1, 2, 3], names: [:data], type: :u8), 1)
      #Nx.Tensor<
        u8[data: 3]
        [1, 2, 3]
      >

      iex> NxSgemm.multiply().(1, Nx.tensor([1, 2, 3], names: [:data], type: :u8))
      #Nx.Tensor<
        u8[data: 3]
        [1, 2, 3]
      >

      iex> NxSgemm.multiply().(Nx.tensor([1.0, 2.0, 3.0], names: [:data], type: :f32), 2.0)
      #Nx.Tensor<
        f32[data: 3]
        [2.0, 4.0, 6.0]
      >

      iex> NxSgemm.multiply().(2.0, Nx.tensor([1.0, 2.0, 3.0], names: [:data], type: :f32))
      #Nx.Tensor<
        f32[data: 3]
        [2.0, 4.0, 6.0]
      >
  """
  def multiply() do
    if SME.available?() and SME.use?() do
      &multiply_sme/2
    else
      &multiply_n/2
    end
  end

  defp multiply_n(a, b) when is_integer(a) and is_integer(b) do
    Nx.tensor(a * b, type: :s32)
  end

  defp multiply_n(a, b) when is_float(b) do
    case Nx.type(a) do
      {:f, 32} ->
        %{
          a
          | data: %{
              a.data
              | state: mul_nif_f32_tensor_f32_scalar(Nx.size(a), a.data.state, b)
            }
        }
    end
  end

  defp multiply_n(a, b) when is_integer(b) when 0 <= b and b < 256 do
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

  defp multiply_n(a, b) when is_number(a) do
    multiply_n(b, a)
  end

  defp multiply_sme(a, b) when is_integer(a) and is_integer(b) do
    Nx.tensor(a * b, type: :s32)
  end

  defp multiply_sme(a, b) when is_float(b) do
    case Nx.type(a) do
      {:f, 32} ->
        %{
          a
          | data: %{
              a.data
              | state: mul_nif_f32_tensor_f32_scalar_sme(Nx.size(a), a.data.state, b)
            }
        }
    end
  end

  defp multiply_sme(a, b) when is_integer(b) when 0 <= b and b < 256 do
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

  defp multiply_sme(a, b) when is_number(a) do
    multiply_sme(b, a)
  end

  defp mul_nif_f32_tensor_f32_scalar(_size, _a, _b),
    do: raise("NIF mul_nif_f32_tensor_f32_scalar/3 not implemented")

  defp mul_nif_f32_tensor_f32_scalar_sme(_size, _a, _b),
    do: raise("NIF mul_nif_f32_tensor_f32_scalar_sme/3 not implemented")

  defp mul_nif_u8_tensor_u8_scalar(_size, _a, _b),
    do: raise("NIF mul_nif_u8_tensor_u8_scalar/3 not implemented")

  @doc """
  Returns the dot product of two tensors.

  Given `a` and `b`, computes the dot product according to the following rules:

  * If both `a` and `b` are scalars, it is equivalent to `a * b`.
  * If `a` is a scalar and `b` is a tensor, it is equivalent to `Nx.multiply(a, b)`.
  * If `a` is a tensor and `b` is a scalar, it is equivalent to `Nx.multiply(a, b)`.
  * If both `a` and `b` are 1-D tensors (vectors), it is the sum of the element-wise product between `a` and `b`. The lengths of `a` and `b` must be equal.
  * If both `a` and `b` are 2-D tensors (matrices), it is equivalent to matrix-multiplication.
  * If either `a` or `b` is a 1-D tensor, and the other is an n-D tensor, it is the sum of the element-wise product along the last axis of `a` or `b`. The length of the 1-D tensor must match the last dimension of the n-D tensor.
  * If `a` is an n-D tensor and `b` is an m-D tensor, it is the sum of the element-wise product along the last axis of `a` and the second-to-last axis of `b`. The last dimension of `a` must match the second-to-last dimension of `b`.

  ## Examples

      iex> left = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      iex> right = Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        f32[2][2]
        [
          [58.0, 64.0],
          [139.0, 154.0]
        ]
      >
  """
  def dot(a, b) do
    case {Nx.type(a), Nx.type(b), Nx.shape(a), Nx.shape(b)} do
      {{:f, 32}, {:f, 32}, {m, n}, {n, o}} ->
        c = Nx.iota({m, o}, type: {:f, 32})

        %{
          c
          | data: %{
              c.data
              | state: dot_nif_f32_matrix_f32_matrix(m, o, n, a.data.state, b.data.state)
            }
        }
    end
  end

  defp dot_nif_f32_matrix_f32_matrix(_m, _o, _n, _a, _b),
    do: raise("NIF dot_nif_f32_matrix_f32_matrix/5 not implemented")
end
