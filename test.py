from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
# jl.using("TextAnalysis")


""" how to call julia methods from python
First, include the file with the methods: jl.eval('include("script.jl")')
If you need to pass data: Main.data = data
To call methods and return values: return_vals = jl.eval("method(data)")

source: https://towardsdatascience.com/how-to-embed-your-julia-code-into-python-to-speed-up-performance-e3ff0a94b6e
"""

#jl.eval('include("/home/kylesa/avast_clf/v0.2/rl_loop.jl")')
#jl.eval("classy()")
#stl = jl.eval("classy()")
#print(stl)


jl.eval('include("/home/kylesa/avast_clf/v0.2/python_utils.jl")')
Main.data = "/home/kylesa/avast_clf/v0.2/data_avast/fffae359332148170c2cad17.json"
stl = jl.eval("classify(data)")
print(stl)
Main.threshold = 0.99
stl = jl.eval("explanation(data,threshold)")
print(stl)