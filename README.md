see the README of the subdirectories for more details.

install python 3.11

install pipenv: `pip install pipenv`

install julia 1.8 (https://julialang.org/downloads/). It has to be available in the path.

install the dependencies by doing:
```sh
pipenv install --dev
pipenv run install
```

run a script by doing:

```sh
pipenv run <script>
```

when adding or removing a package to the Julia project, regenerate the manifest with Pkg.resolve()