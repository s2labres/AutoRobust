using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0
Pkg.offline()
Pkg.activate("/home/kylesa/avast_clf/v0.2/")
Pkg.develop(PackageSpec(path = "/home/kylesa/avast_clf/ExplainMill_example/ExplainMill.jl-master"))
using Flux, MLDataPattern, Mill, JsonGrinder, JSON, Statistics, IterTools, StatsBase, ThreadTools
using JsonGrinder: suggestextractor, ExtractDict
using Mill: reflectinmodel
using CSV, DataFrames
using Random
# using PrettyTables
using ExplainMill
using Printf
using PyCall
# using Plots
# using Dates
using BSON: @load
using BSON: @save

ENV["JULIA_NUM_THREADS"] = 32
Threads.nthreads() = 32

base_path = "/home/kylesa/avast_clf/v0.2/"
Settings = Dict("model_multi_path"=>base_path*"model_multi.bson",
"extractor_multi_path"=>base_path*"extractor_multi.bson",
"model_adv_path"=>base_path*"model_adv.bson",
"extractor_adv_path"=>base_path*"extractor_adv.bson",
"labels"=>base_path*"data_avast/labels.csv",
"adv_labels"=>base_path*"data_smp/labels_adv.csv",
"report_folder"=>base_path*"data_avast/",
"adv_folder"=>base_path*"data_smp")

JsonGrinder.skip_single_key_dict!(false) # DO NOT REMOVE ME

function load_model(advtrain)
    if advtrain
        @load Settings["model_adv_path"] model
        @load Settings["extractor_adv_path"] extractor
        print("Loading adv model...\n")
    else
        @load Settings["model_multi_path"] model
        @load Settings["extractor_multi_path"] extractor
        print("Loading norm model...\n")
    end
    return model,extractor
end

function keys_to_string(dict::Dict{Any, Any})
    new_dict = Dict{String, Any}()
    for (key, value) in dict
        if isa(key, Dict)
            new_key = keys_to_string(key)
        else
            new_key = string(key)
        end
        new_dict[new_key] = value
    end
    return new_dict
end

function explanation(sample,threshold)
    ds = extractor(sample; store_input=true)
    e = ExplainMill.DafExplainer()
    explanation = explain(e, ds, model; rel_tol=threshold, pruning_method=:LbyL_HArr)
    rule = e2boolean(ds,explanation,extractor)
    return rule
end

function split_train_test(d,pct)
    test = Dict()
    train = Dict()
    for (key,value) in d
        train_,test_ = splitobs(value,pct)
        #println("$(length(train_)) $(length(test_)) $(length(value))")
        train[key] = train_
        test[key] = test_
    end
    return train,test
end

function classify(report)
    return softmax(model(extractor(report; store_input=true)).data), 0
end

function classify2(report)
    # Moved load model out of function
    extracted_sample = extractor(report; store_input=true)
    # embedding = vcat(model.ms[:static](rep[:static]), model.ms[:behavior](rep[:behavior]))
    emb = model.ms[:summary](extracted_sample[:summary])
    # model.m(emb)
    emb_val = softmax(emb.data)
    output = softmax(model.m(emb).data)
    return output, emb_val
end

function loadrep()
    global report = open(JSON.parse, base_path*"curradv.json")
end

function retrain(balanced=false)
    df_labels = dfl
    jsons = jsn
    train_indexes = tdx
    test_indexes = rdx
    adv_labels = CSV.read(Settings["adv_labels"], DataFrame)

    indices = Dict()
    for (i,v) in enumerate(adv_labels.family)
        if haskey(indices,v)
            push!(indices[v],i)
        else
            indices[v] = [i]
        end
    end
    adv_train, adv_test = split_train_test(indices,0.8)
    atrain_indexes = []
    atest_indexes = []
    for (k,v) in adv_train
        atrain_indexes = vcat(atrain_indexes,v)
    end
    for (k,v) in adv_test 
        atest_indexes= vcat(atest_indexes,v)
    end

    ADVS_PATH = Settings["adv_folder"]
    advs = tmap(adv_labels.hash) do s
        try 
            open(JSON.parse, "$(ADVS_PATH)/$(s)_adv.json")
        catch e
            @error "Error when processing sha $s: $e"
        end
    end ;

    sz = length(jsons)
    clean = deepcopy(jsons)

    # Append adv reports to ben array
    for elem in advs
        append!(jsons, [elem])
    end

    # Append adv labels to original
    mx_labels = vcat(df_labels, adv_labels)

    atrain_indexes = atrain_indexes .+ sz
    atest_indexes = atest_indexes .+ sz
    total_train = vcat(train_indexes, atrain_indexes)
    println(length(total_train))
    println(length(atrain_indexes))

    chunks = Iterators.partition(total_train, 28)
    sch_parts = tmap(chunks) do ch
        JsonGrinder.schema(jsons[ch])
    end

    # load model and extractor
    model,extractor = load_model(advflag)
    data = tmap(extractor, jsons) ;
    cdata = tmap(extractor, clean) ;
    # println(size(data, 1))
    # println(size(mx_labels, 1))
    @assert size(data, 1) == size(mx_labels, 1)
    labelnames = sort(unique(mx_labels.family))
   
    num_epochs = 1
    minibatchsize = 128
    iterations = ceil(Int, num_epochs * (length(total_train) / minibatchsize))

    function minibatch()
        idx = sample(total_train, minibatchsize, replace = false)
        reduce(catobs, data[idx]), Flux.onehotbatch(mx_labels.family[idx], labelnames)
    end

    function accuracy(x,y) 
        vals = tmap(x) do s
            Flux.onecold(softmax(model(s).data), labelnames)[1]
        end
        mean(vals .== y)
    end
    
    # eval_trainset = shuffle(train_indexes)[1:1000]
    # eval_testset = shuffle(test_indexes)[1:1000]
    # eval_advset = shuffle(atest_indexes)[1:200]
    # cb = () -> begin
    #     train_acc = accuracy(data[eval_trainset], mx_labels.family[eval_trainset])
    #     test_acc = accuracy(data[eval_testset], mx_labels.family[eval_testset])
    #     robust_acc = accuracy(data[eval_advset], mx_labels.family[eval_advset])
    #     println("accuracy: train = $train_acc, test = $test_acc, adv = $robust_acc")
    # end
    ps = Flux.params(model)
    loss = (x,y) -> Flux.logitcrossentropy(model(x).data, y)
    opt = ADAM()
    # opt = ADAM(0.01) custom learning rate

    # train
    # Flux.Optimise.train!(loss, ps, repeatedly(minibatch, iterations), opt, cb = Flux.throttle(cb, 2))
    Flux.Optimise.train!(loss, ps, repeatedly(minibatch, iterations), opt)
    clean_accuracy = accuracy(cdata[test_indexes], df_labels.family[test_indexes])
    robust_accuracy = accuracy(data[atest_indexes], mx_labels.family[atest_indexes])
    println("Final evaluation:")
    println("Clean accuracy on test data: $(clean_accuracy)")
    println("Robust accuracy on test data: $(robust_accuracy)")
    test_predictions = Dict()
    # true_label = labelnames[1]
    for true_label in labelnames
        current_predictions = Dict()
        [current_predictions[pl]=0.0 for pl in labelnames]
        family_indexes = filter(i -> df_labels.family[i] == true_label, test_indexes)
        predictions = tmap(data[family_indexes]) do s
            Flux.onecold(softmax(model(s).data), labelnames)[1]
        end
        [current_predictions[pl] += 1.0 for pl in predictions]
        [current_predictions[pl] = current_predictions[pl] ./ length(predictions) for pl in labelnames]
        test_predictions[true_label] = current_predictions
    end
    @printf "%8s\t" "TL\\PL"
    [@printf " %8s" s for s in labelnames]
    print("\n")
    for tl in labelnames
        @printf "%8s\t" tl 
        for pl in labelnames
            @printf "%9s" @sprintf "%.2f" test_predictions[tl][pl]*100
        end
        print("\n")
    end
    @save Settings["model_adv_path"] model
    @save Settings["extractor_adv_path"] extractor
    return clean_accuracy, robust_accuracy
end

function load_data(balanced=false)
    df_labels = CSV.read(Settings["labels"], DataFrame)
    indexes = Dict()
    for (i,v) in enumerate(df_labels.family)
        if haskey(indexes,v)
            push!(indexes[v],i)
        else
            indexes[v] = [i]
        end
    end
    train,test = split_train_test(indexes,0.8)

    if balanced
        lowebound = minimum([v for (k,v) in countmap(df_labels.family)])
        print(lowebound)
        for (k,v) in train 
            if length(v) > lowebound
                train[k] = v[1:lowebound]
            end
        end
    end
    train_indexes = []
    test_indexes = []
    for (k,v) in train
        train_indexes = vcat(train_indexes,v)
    end
    for (k,v) in test 
        test_indexes= vcat(test_indexes,v)
    end

    # load json reports
    JSONS_PATH = Settings["report_folder"]
    jsons = tmap(df_labels.hash) do s
        try 
            open(JSON.parse, "$(JSONS_PATH)/$(s).json")
        catch e
            @error "Error when processing sha $s: $e"
        end
    end
    return df_labels, jsons, train_indexes, test_indexes
end

# load model and extractor
model,extractor = load_model(advflag)

# load benign data
# if !(@isdefined dfl)
a, b, c, d = load_data()
const dfl = a
const jsn = b
const tdx = c
const rdx = d
# end