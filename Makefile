file_path := $(abspath $(lastword $(MAKEFILE_LIST)))
file_dir := $(dir $(file_path))
source_dir := $(file_dir)docs
build_dir := $(file_dir)public

.PHONY : doc clean clean_doc

# Remove all python generated files
clean:
	-find . -name '*.py[co]' -exec rm {} +
	-find . -name '__pycache__' -exec rm -rf {} +
	
# Remove all sphinx generated files
clean_doc:
	-rm -rf ./public
	-rm -rf ./docs/reference/api

# Generate the documentation
doc: clean clean_doc
	cd docs; \
	sphinx-build -W -b html $(source_dir) $(build_dir)
