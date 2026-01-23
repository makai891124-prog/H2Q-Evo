# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line. 
SPHINXOPTS    = -W -n
SPHINXBUILD   = sphinx-build
SPHINXSOURCEDIR = docs/source
SPHINXOUTPUTDIR = docs/build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SPHINXSOURCEDIR)" "$(SPHINXOUTPUTDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%	:
	@$(SPHINXBUILD) -M $@ "$(SPHINXSOURCEDIR)" "$(SPHINXOUTPUTDIR)" $(SPHINXOPTS) $(O)