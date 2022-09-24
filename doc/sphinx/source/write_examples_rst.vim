call append(0, ["Tutorials & Examples",
	       \"====================",
	       \"",
	       \".. toctree::",
	       \"   :maxdepth: 1",
	       \])
norm G
read !ls -1 examples/*.ipynb
g|^examples/0\+_|d
g/^examples/norm I   
write! examples.rst
quit
