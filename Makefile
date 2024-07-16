file_path := $(abspath $(lastword $(MAKEFILE_LIST)))
file_dir := $(dir $(file_path))
source_dir := $(file_dir)docs
build_dir := $(file_dir)public

.PHONY : docs clean clean_docs clean_notebooks copy_notebooks generate_docs copy_binary_outputs

# Remove all python generated files
clean:
	-find . -name '*.py[co]' -exec rm {} +
	-find . -name '__pycache__' -exec rm -rf {} +
	
# Remove all sphinx generated files
clean_docs:
	-rm -rf ./public
	-rm -rf ./docs/reference/api

# Remove any notebooks from docs
clean_notebooks:
	-rm -rf ./docs/example_notebooks/notebooks

# Copy notebooks
copy_notebooks: 
	-mkdir ./docs/example_notebooks/notebooks
	-cp -R ./example-notebooks/* ./docs/example_notebooks/notebooks

# Generate documentation
generate_docs:
	cd docs; \
	sphinx-build -W -b html $(source_dir) $(build_dir)

copy_binary_outputs:
	-mkdir ./public/example_notebooks/notebooks/binary-classifier/outputs
	-cp ./docs/example_notebooks/notebooks/binary-classifier/outputs/* ./public/example_notebooks/notebooks/binary-classifier/outputs

# Put it all together
docs: clean clean_docs clean_notebooks copy_notebooks generate_docs copy_binary_outputs
