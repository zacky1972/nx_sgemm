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
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
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
end
