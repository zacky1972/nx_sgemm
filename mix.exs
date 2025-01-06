defmodule NxSgemm.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_sgemm,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:ex_task] ++ Mix.compilers(),

      # Docs
      name: "NxSgemm",
      source_url: "https://github.com/zacky1972/nx_sgemm",
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    System.put_env("SME_AVAILABLE", to_string(sme_runnable?() and sme_compilable?()))

    [
      extra_applications: [:logger],
      mod: {NxSgemm.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:autoconfex, "~> 0.1"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:ex_task, "~> 0.1", runtime: false},
      {:nx, "~> 0.9"}
    ]
  end

  defp docs do
    [
      main: "NxSgemm",
      extras: ["README.md", "LICENSE"]
    ]
  end

  defp sme_runnable?() do
    case :os.type() do
      {:unix, :darwin} -> sme_runnable_s?()
      _ -> false
    end
  end

  defp sme_runnable_s?() do
    case sme_execute("sysctl", ["hw.optional.arm"]) do
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

  defp sme_execute(executable, options) do
    case System.find_executable(executable) do
      nil -> false
      executable -> System.cmd(executable, options)
    end
  end

  defp sme_compilable?() do
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
