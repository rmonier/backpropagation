see the README of the subdirectories for more details.

julia has to be installed and available in the path.
install by doing:

pipenv install --dev

run a script by doing:

pipenv run <script>


when adding or removing a package to the Julia project, regenerate the manifest with Pkg.resolve()