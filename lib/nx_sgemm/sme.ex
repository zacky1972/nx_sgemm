defmodule SME do
  @moduledoc """
  Module for Scalable Matrix Extension (SME).
  """
  def available?() do
    {:ok, pid} = Task.Supervisor.start_link()

    task1 = Task.Supervisor.async(pid, fn -> runnable?() end)
    task2 = Task.Supervisor.async(pid, fn -> compilable?() end)

    Task.await(task1) and Task.await(task2)
  end

  def runnable?() do
    case :os.type() do
      {:unix, :darwin} -> runnable_s?()
      _ -> false
    end
  end

  defp runnable_s?() do
    case execute("sysctl", ["hw.optional.arm"]) do
      {result, 0} ->
        result
        |> String.split("\n")
        |> Enum.map(&String.trim/1)
        |> Enum.filter(&String.match?(&1, ~r/FEAT\_SME/))
        |> Enum.map(&String.split(&1, " "))
        |> Enum.count(fn [_, v] -> v == "1" end)
        |> then(&(&1 != 0))

      _ ->
        false
    end
  end

  defp execute(executable, options) do
    case System.find_executable(executable) do
      nil -> false
      executable -> System.cmd(executable, options)
    end
  end

  def compilable?() do
    Autoconfex.compilable_by_cc?(
      "/usr/bin/clang",
      """
      #include <arm_sme.h>

      __arm_locally_streaming
      __arm_new("za")
      void test_arm_new(void) {}
      int main(int argc, char *argv[])
      {
        test_arm_new();
      }
      """,
      ["-O2", "-march=armv9-a+sme"]
    )
  end
end
