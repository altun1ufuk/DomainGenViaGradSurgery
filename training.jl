include("preprocess.jl");

function train(mode, dataset, target, model; num_iter=1000, check_freq=50, batchsize=128, data_path="/Users/ufukaltun/Documents/koç/dersler/ku deep learning/project/data", atype=Array{Float32})


if dataset == "PACS"
    d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3 = PACS(data_path, batchsize, atype, target);
    println("Datasets are loaded")

    if mode == "b"

          tst_loss, tst_acc = train_b(model, d_trn, d_val, d_tst, check_freq, num_iter, target)

    elseif mode == "gs"

          tst_loss, tst_acc = train_gs(model, d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3, check_freq, num_iter, target)
    end
elseif dataset == "VLCS"
    d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3 = VLCS(data_path, batchsize, atype, target);
    println("Datasets are loaded")

    if mode == "b"

          tst_loss, tst_acc = train_b(model, d_trn, d_val, d_tst, check_freq, num_iter, target)

    elseif mode == "gs"

          tst_loss, tst_acc = train_gs(model, d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3, check_freq, num_iter, target)
    end
else
    println("Invalid dataset")
end
    return tst_loss, tst_acc
end



function train_b(model, d_trn, d_val, d_tst, check_freq, num_iter, target)
    n=1
    results=[]
    best_acc=0
    for batch in progress(cycle(d_trn))
        D = @diff model(batch...);
        for w in Knet.params(model)
            g = grad(D, w)
            update!(w, g)
        end
        if n==1 
            println("First batch is trained successfully")
            best_acc = accuracy(model,data=d_val)
        end
        if n%check_freq == 0
            acc = accuracy(model,data=d_val)
            if acc > best_acc
                best_acc = acc
                save(string("best_model_b_target_", target,".jld2"),"weights",model)
            end
            result=(model(d_trn), model(d_val), 
                        accuracy(model,data=d_trn), acc)
            println("Train loss: ",result[1],"  Val loss: ",result[2],
                    "  Train acc: ",result[3],"  Val acc: ",result[4])
            push!(results,result)
        end

        (n += 1) > num_iter && break
    end 
    println("Training ended successfully, saving the results")
    save(string("results_b_target_", target,".jld2"),"results",results)
    best_model = load(string("best_model_b_target_", target,".jld2"))["weights"]
    tst_loss = best_model(d_tst)
    tst_acc = accuracy(best_model,data=d_tst)
    println("Test loss: ", tst_loss, "  Test acc: ", tst_acc)
    return tst_loss, tst_acc
end


function train_gs(model, d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3, check_freq, num_iter, target)
    n=1
    results=[]
    best_acc=0
    for (dom1, dom2, dom3) in progress(zip(cycle(d_trn_dom1), cycle(d_trn_dom2), cycle(d_trn_dom3)))
        D1 = @diff model(dom1...)
        D2 = @diff model(dom2...)
        D3 = @diff model(dom3...)
        
        for w in Knet.params(model)
            g1 = grad(D1, w)
            g2 = grad(D2, w)
            g3 = grad(D3, w)
            MaskGS = abs.(sign.(g1).+sign.(g2).+sign.(g3)).==3
            g = (g1.+g2.+g3).*MaskGS
            update!(w, g)
        end
        
        if n==1 
            println("First batch is trained successfully")
            best_acc = accuracy(model,data=d_val)
        end
        
        if n%check_freq == 0
            acc = accuracy(model,data=d_val)
            if acc > best_acc
                best_acc = acc
                save(string("best_model_gs_target_", target,".jld2"),"weights",model)
            end
            result=(model(d_trn), model(d_val), 
                        accuracy(model,data=d_trn), acc)
            println("Train loss: ",result[1],"  Val loss: ",result[2],
                    "  Train acc: ",result[3],"  Val acc: ",result[4])
            push!(results,result)
        end
        
        (n += 1) > num_iter && break
    end 
    println("Training ended successfully, saving the results")
    save(string("results_gs_target_", target,".jld2"),"results",results)
    best_model = load(string("best_model_gs_target_", target,".jld2"))["weights"]
    tst_loss = best_model(d_tst)
    tst_acc = accuracy(best_model,data=d_tst)
    println("Test loss: ", tst_loss, "  Test acc: ", tst_acc)
    return tst_loss, tst_acc
end


