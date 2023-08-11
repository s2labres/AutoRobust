using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0
Pkg.offline()
Pkg.activate("./")
Pkg.develop(PackageSpec(path = "/home/kylesa/avast_clf/ExplainMill_example/ExplainMill.jl-master"))
import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("Plots")
import Pkg; Pkg.add("BenchmarkTools")

using Flux, MLDataPattern, Mill, JsonGrinder, JSON, Statistics, IterTools, StatsBase, ThreadTools
using JsonGrinder: suggestextractor, ExtractDict
using Mill: reflectinmodel
using CSV, DataFrames
using Random
using PrettyTables
using ExplainMill
using BenchmarkTools
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

function load_model()
    if isfile(Settings["model_multi_path"]) && isfile(Settings["extractor_multi_path"])
        print("Loading the model\n")
        @load Settings["model_multi_path"] model
        @load Settings["extractor_multi_path"] extractor
        # df_labels = CSV.read(Settings["labels"], DataFrame)
        # print(df_labels[5000:5100,:])
    else
        print("No model found, create one ... ")
    end
    return model,extractor
end

function classify(model,extractor,sample)
    return softmax(model(extractor(sample; store_input=true)).data) 
end

# function

function perturb(data)
    for (key,value) in data
        #print(typeof(value))
        if key == "_" || key == "_"
            continue
        else
            if typeof(value) == Vector{Any}
                v = Any[]
                for i = 1:rand(1:20)
                    push!(v,randstring(rand(10:30)))
                end
                # print(v)
                data[key] = v
                # for (i, v) in pairs(value)
                #     # if size(split(v,"\\"))[1] > 1
                #     #     original = split(v,"\\")[size(split(v,"\\"))[1]-1]*"\\"*split(v,"\\")[size(split(v,"\\"))[1]]
                #     #     new_value = replace(v,original=>randstring(5)*"\\"*randstring(5))
                #     # else
                #     #     new_value = replace(v,split(v,"\\")[size(split(v,"\\"))[1]]=>randstring(5)*"\\"*randstring(5))
                #     # end
                #     # value[i] = new_value
                #     value[i] = randstring(15)
                # end
                # data[key] = value
            elseif typeof(value) == Dict{String,Any}
                perturb(value)
            else
                # if size(split(value,"\\"))[1] > 1
                #     original= split(value,"\\")[size(split(value,"\\"))[1]-1]*"\\"*split(value,"\\")[size(split(value,"\\"))[1]]
                #     new_value = replace(value,original=>randstring(5)*"\\"*randstring(5))
                # else
                #     new_value = replace(value,split(value,"\\")[size(split(value,"\\"))[1]]=>randstring(5)*"\\"*randstring(5))
                # end
                # data[key] = new_value
                data[key] = randstring(20)
            end
        end
    end
end

function extend(data, source)
    for (key,value) in data["summary"]
        if key == "started_services" || key == "created_services" || key == "delete_files" || key == "delete_keys"
            continue
        else
            data["summary"][key] = reduce(vcat, [data["summary"][key], source[key]])
        end
    end
end

function explanation(model,extractor,sample,value)
    ds = extractor(sample; store_input=true)
    e = ExplainMill.DafExplainer()
    explanation = explain(e, ds, model; rel_tol=value, pruning_method=:LbyL_HArrft)
    rule = e2boolean(ds,explanation,extractor)
    return rule
end

# function explanation(model,value,ex)
#     explanation = explain(ex, ds, model; rel_tol=value, pruning_method=:LbyL_HArr)
# end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "report"
            help = "target report"
            required = true
        "threshold"
            help = "class threshold"
            default = 0.5
    end

    return parse_args(s)
end

function get_elements(data)
    to_ret = Vector{String}()
    if typeof(data) == Vector{String}
        for j in data
            push!(to_ret,j)
        end
    elseif  typeof(data) == Dict{Symbol,Vector{String}}
        for (key,value) in data
            if  typeof(value) == Dict{Symbol,Vector{String}} || typeof(value) == Vector{String}
                to_add = get_elements(value)
                while typeof(to_add) == Vector{String}
                    to_add = first(to_add)
                end
                push!(to_ret,to_add)
            else
                push!(to_ret,value)
            end
        end
    else
        push!(to_ret,data)
    end
    return to_ret
end

function flat(data)
    to_ret = Vector{String}()
    for element in data
        if typeof(data) == Vector{String}
            for item in data
                if typeof(item) == Vector{String}
                    for iitem in item
                        push!(to_ret,iitem)
                    end
                else
                    push!(to_ret,item)
                end
            end
        else
            push!(to_ret,element)
        end
    end
    return to_ret
end

function classy()
    model,extractor = load_model()
    target_report = JSON.parsefile("/home/kylesa/avast_clf/v0.2/data_avast/0b92f001087f1a9399f62981.json")
    classification = classify(model,extractor,target_report)
    print("Current classification $(classification)\n")
    return 5
end


function main()
    # parsed_args = parse_commandline()
    # virlock
    # parsed_args = Dict("report"=>"/home/kylesa/avast_clf/v0.2/data_avast/0a1be75535f39fda79705c14.json", "threshold"=>"0.3")
    # shipup
    # parsed_args = Dict("report"=>"/home/kylesa/avast_clf/v0.2/data_avast/0112a871fe8ada279cda4358.json", "threshold"=>"0.3")
    # vobfus
    parsed_args = Dict("report"=>"/home/kylesa/avast_clf/v0.2/data_avast/0b92f001087f1a9399f62981.json", "threshold"=>"0.3")
    target_report = JSON.parsefile(parsed_args["report"])
    source = JSON.parsefile("/home/kylesa/avast_clf/v0.2/sum.json")
    model, extractor = load_model()
    evaded = false
    initial = nothing
    iteration = 0
    while !evaded
        classification = classify(model,extractor,target_report)
        exp = explanation(model,extractor,target_report,0.99)
        print("Current classification $(classification)\n")
        classification = classification .> parse(Float32,parsed_args["threshold"])
        if initial == nothing
            initial = classification
            print("Starting label $(classification)\n")
        elseif initial != classification
            print("Evaded!")
            evaded = true
        else
            # guide = explanation(model,extractor,target_report,explainer_value)
            # print(guide)
            # perturb
            # perturb(target_report)
            extend(target_report, source)  
        end
        iteration += 1
        print("Current iteration: $(iteration)\n")
        open("ptb.json","w") do f
            JSON.print(f,target_report) 
        end
        if iteration >= 10
            break
        end
    end
    open(base_path*"/reports/$(last(split(parsed_args["report"],"/")))","w") do f 
        JSON.print(f, target_report)
    end
end

main()

# parsed_args = Dict("report"=>"/home/kylesa/avast_clf/v0.2/data_avast/0b92f001087f1a9399f62981.json", "threshold"=>"0.3")
# target_report = JSON.parsefile(parsed_args["report"])
# source = JSON.parsefile("/home/kylesa/avast_clf/v0.2/sum.json")
# model, extractor = load_model()
# tresh = 0.99
# ds = extractor(target_report; store_input=true)
# ex = ExplainMill.DafExplainer()
# while true
#     @btime explanation(model,tresh,ex)
# end