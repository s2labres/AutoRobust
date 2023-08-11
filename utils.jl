function extract_common(dicts)
    # initialize dict
    # d = Dict(name => Any[]) for name in data[0]
    dict = Dict()
    for dict in dicts
        # Iterate over the keys in the current dictionary
        for (key, values) in dict
            # If the key doesn't exist in the result dictionary yet, add it with an empty array
            if !haskey(result_dict, key)
                result_dict[key] = []
            end
            # Append the values in the current dictionary to the array in the result dictionary
            push!(result_dict[key], values...)
            # append!(result_dict[key], values...)
        end
    end
    return dict
end

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