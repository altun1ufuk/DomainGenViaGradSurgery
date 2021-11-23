# For running experiments
function trainresultsGS(file,model; o...)
    if (print("Train from scratch? "); readline()[1]=='y')
        r = ((model(dtrn_baseline), model(dtst), 1-accuracy(model,dtrn_baseline), 1-accuracy(model,dtst))
             for x in takenth(progress(sgdGS(model,ncycle(dtrn1,10),ncycle(dtrn2,10),ncycle(dtrn3,10))),length(dtrn1)))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        save(file,"results",r)
        GC.gc(true) # To save gpu memory
    else
        isfile(file) || download("https://github.com/denizyuret/Knet.jl/releases/download/v1.4.9/$file",file)
        r = load(file,"results")
    end
    println(minimum(r,dims=2))
    return r
end

# For running experiments
function trainresults(file,model; o...)
    if (print("Train from scratch? "); readline()[1]=='y')
        r = ((model(dtrn_baseline), model(dtst), 1-accuracy(model,dtrn_baseline), 1-accuracy(model,dtst))
             for x in takenth(progress(sgd(model,ncycle(dtrn_baseline,10))),length(dtrn_baseline)))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        save(file,"results",r)
        GC.gc(true) # To save gpu memory
    else
        isfile(file) || download("https://github.com/denizyuret/Knet.jl/releases/download/v1.4.9/$file",file)
        r = load(file,"results")
    end
    println(minimum(r,dims=2))
    return r
end