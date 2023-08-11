using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0
Pkg.offline()
Pkg.activate("./")
Pkg.develop(PackageSpec(path = "/home/kylesa/avast_clf/ExplainMill_example/ExplainMill.jl-master"))
import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("Plots")

using Flux, MLDataPattern, Mill, JsonGrinder, JSON, Statistics, IterTools, StatsBase, ThreadTools
using JsonGrinder: suggestextractor, ExtractDict
using Mill: reflectinmodel
using CSV, DataFrames
using Random
using PrettyTables
using ExplainMill
using Printf
using Plots
using Dates
using BSON: @load
using BSON: @save
using Random
using ArgParse
base_path = "/home/kylesa/avast_clf/v0.2/"
cd(base_path)
include("utils.jl")

Settings = Dict("model_multi_path"=>base_path*"model_multi.bson",
"extractor_multi_path"=>base_path*"extractor_multi.bson",
"labels"=>base_path*"data_avast/labels.csv",
"report_folder"=>base_path*"data_avast/")

JsonGrinder.skip_single_key_dict!(false)

keyser = ["keys",
 "resolved_apis",
 "executed_commands",
 "write_keys",
 "files",
 "read_files",
#  "started_services",
#  "created_services",
 "write_files",
#  "delete_keys",
 "read_keys",
#  "delete_files",
 "mutexes"]

 maxval = 3

function get_most_frequent(reports, maxval)
    sumrep = Dict()
    for report in reports
        trep = JSON.parsefile(report)
        # Iterate over the keys in the current dictionary
        for (key, values) in trep["summary"]
            # If the key doesn't exist in the result dictionary yet, add it with an empty array
            if !haskey(sumrep, key)
                sumrep[key] = []
            end
            # Append the values in the current dictionary to the array in the result dictionary
            push!(sumrep[key], values...)
            # append!(result_dict[key], values...)
        end
    end

    freqrep = Dict()
    for keys in keyser
        freq = countmap(sumrep[keys])
        # Get the values of the dictionary as an array
        vl = collect(values(freq))
        sorted_values = sort(vl, rev=true)
        top_values = sorted_values[1:maxval]
        print(keys, sorted_values)
        top_set = Set(top_values)
        # print(top_values, top_set)
        top_keys = []

        # Iterate over the key-value pairs in the dictionary
        for (key, value) in freq
            # If the current value is in the top values, append the key to the top_keys array
            if value in top_set
                push!(top_keys, key)
            end
        end

        freqrep[keys] = top_keys

        # print(top_keys)
    end
        
    open("sum.json","w") do f
        JSON.print(f,freqrep) 
    end
end

function main()
    # reports = ["/home/kylesa/avast_clf/v0.2/data_avast/0a1be75535f39fda79705c14.json",
    #          "/home/kylesa/avast_clf/v0.2/data_avast/0b92f001087f1a9399f62981.json",
    #          "/home/kylesa/avast_clf/v0.2/data_avast/0112a871fe8ada279cda4358.json"]
    base = "/home/kylesa/avast_clf/v0.2/data_avast/"

    df_labels = CSV.read(Settings["labels"], DataFrame)
    # print(df_labels[family=="shipup"])
    df = filter(row -> row.family == "virlock", df_labels)
    # print(df[1:10,1][1])
    # df[1:10,1]
    files = [base*v*".json" for v in df[:,1]]
    # print(files)
    # get_most_frequent(report, maxval)
    get_most_frequent(files,10)
end

main()