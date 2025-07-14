# Creating this package

## PkgTemplates

```
Template:
  authors: ["Jeffrey Wack <jeffwack111@gmail.com> and contributors"]
  dir: "~/.julia/dev"
  host: "github.com"
  julia: v"1.11.0"
  user: "jeffwack"
  plugins:
    Citation:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/CITATION.bib"
      readme: true
    Dependabot:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/github/dependabot.yml"
    Documenter:
      assets: String[]
      logo: Logo(nothing, nothing)
      makedocs_kwargs: Dict{Symbol, Any}()
      canonical_url: PkgTemplates.github_pages_url
      make_jl: "~/.julia/packages/PkgTemplates/5mBnk/templates/docs/make.jlt"
      index_md: "~/.julia/packages/PkgTemplates/5mBnk/templates/docs/src/index.md"
      devbranch: nothing
      edit_link: :devbranch
    Git:
      ignore: String[]
      name: nothing
      email: "jeffwack111@gmail.com"
      branch: "main"
      ssh: false
      jl: true
      manifest: false
      gpgsign: false
    GitHubActions:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/github/workflows/CI.yml"
      destination: "CI.yml"
      linux: true
      osx: true
      windows: true
      x64: true
      x86: true
      coverage: true
      extra_versions: ["1.6", "1.11", "pre"]
    License:
      path: "~/.julia/packages/PkgTemplates/5mBnk/templates/licenses/MIT"
      destination: "LICENSE"
    ProjectFile:
      version: v"1.0.0-DEV"
    Readme:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/README.md"
      destination: "README.md"
      inline_badges: false
      badge_order: DataType[Documenter{GitHubActions}, Documenter{GitLabCI}, Documenter{TravisCI}, GitHubActions, GitLabCI, TravisCI, AppVeyor, DroneCI, CirrusCI, Codecov, Coveralls, BlueStyleBadge, ColPracBadge, PkgEvalBadge]
      badge_off: DataType[]
    SrcDir:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/src/module.jlt"
    TagBot:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/github/workflows/TagBot.yml"
      destination: "TagBot.yml"
      trigger: "JuliaTagBot"
      token: Secret("GITHUB_TOKEN")
      ssh: Secret("DOCUMENTER_KEY")
      ssh_password: nothing
      changelog: nothing
      changelog_ignore: nothing
      gpg: nothing
      gpg_password: nothing
      registry: nothing
      branches: nothing
      dispatch: nothing
      dispatch_delay: nothing
    Tests:
      file: "~/.julia/packages/PkgTemplates/5mBnk/templates/test/runtests.jlt"
      project: false
      aqua: false
      aqua_kwargs: NamedTuple()
      jet: false
```

## Github Pages

Go to repository settings, deploy from a branch, gh-pages /root.

Now the docs appear on the dev branch. 
