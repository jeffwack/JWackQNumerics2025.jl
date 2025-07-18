# qnumerics.org session on Stabilizer formalism

## Content

1. `clifford.jl` - an introduction to the Stabilizer (and Destabilizer) formalisms for representing certain quantum states and dynamics efficiently on a classical computer.

## Setting up your environment

We suggest using:

- Julia installed through the `juliaup` version multiplexer/manager.
- VS Code (or compatible) IDE in which this folder is opened as a root workspace folder.
- The Julia plugin for VS Code.

Make sure you are comfortable with the "Command Palette" in VS Code,
and have the Julia REPL launched from it.
That provides a better, more integrated user experience,
than simply launching Julia from the terminal.

Do not run entire files, rather familiarize yourself with how to interactively launch
only small snippets of code in the REPL.

If you prefer a notebook interface, you can use Jupyter or Pluto.
Pluto is a particularly elegant, lightweight, reactive notebook server for Julia.

## What are all these other files?

- `Project.toml` lists all the dependencies for this environment. Make sure to `activate` it and `instantiate` it.
- `Manifest.toml` would exist only if you have instantiated the environment. It lists the exact versions of each installed library and implicit dependency, providing complete reproducibility (unlike just a list of library names in `Project.toml`).
- `.gitignore` is there to tell `git` that we do not care to track certain files.
- The `.vscode` folder has some vscode configuration options that set up your IDE in a more comfortable fashion. Chiefly, it makes sure that the activated environment is the local one. Usually `.vscode` is added to the `.gitignore` list.
- The `rendered` folder has all of the codefiles converted to notebooks and markdown files, if you prefer to view them in a different way. This was done using `import Literate; Literate.notebook(filename, "./rendered/";)`. The `Literate` package provides for a convenient way to convert a script into a more pleasant-to-read format.