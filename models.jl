ENV["COLUMNS"]=72
using Base.Iterators: flatten, take, cycle
using IterTools: ncycle, takenth
using Statistics: mean
using CUDA: CUDA, CuArray # functional
using Knet: Knet, conv4, pool, mat, nll, accuracy, progress, sgd, param, param0, dropout, relu, training, minibatch, Data
using Knet, IterTools
using JLD2

# Definition of a convolutional layer (without pooling):
struct Conv1; w; b; f; pd; p; s; end
(c1::Conv1)(x) = c1.f.(conv4(c1.w, dropout(x,c1.pd); padding=c1.p, stride=c1.s) .+ c1.b)
Conv1(w1::Int,w2::Int,cx::Int,cy::Int,f=relu; pdrop=0, padding=0, stride=1)= Conv1(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop, padding, stride)

# Definition of a convolutional layer (with pooling):
struct Conv2; w; b; f; pd; p; s; pw; ps end
(c2::Conv2)(x) = c2.f.(pool(conv4(c2.w, dropout(x,c2.pd); padding=c2.p, stride=c2.s) .+ c2.b; window=c2.pw, stride=c2.ps))
Conv2(w1::Int,w2::Int,cx::Int,cy::Int,f=relu; pdrop=0, padding=0, stride=1, pwindow=3, pstride=2)= Conv2(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop, padding, stride, pwindow, pstride)

# Define a convolutional layer:
struct Conv; w; b; f; p; pad_conv; stri_conv; pad_pool; stri_pool; wind; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p); padding=c.pad_conv, stride=c.stri_conv) .+ c.b ;
        window=c.wind, padding=c.pad_pool, stride=c.stri_pool )) 
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0,pad_conv=(0,0),stri_conv=0,pad_pool=(0,0),stri_pool=0, wind=3)= Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop ,pad_conv,stri_conv,pad_pool,stri_pool, wind)

"""
# Definition of a MaxPooling layer:
struct Pooling; w; s; end
(p::Pooling)(x) = pool(x; window=p.w, stride=p.s )
Pooling(window=3, stride=2) = Pooling(window, stride)
"""

# Definition of a Dense layer:
struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b)
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)



# Let's define a chain of layers
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

# We add two new fields for L1 and L2 regularization
struct Chain2
    layers; λ1; λ2
    Chain2(layers...; λ1=0, λ2=0) = new(layers, λ1, λ2)
end

# The prediction and average loss do not change
(c::Chain2)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain2)(d::Data) = mean(c(x,y) for (x,y) in d)

# The loss function penalizes the L1 and/or L2 norms of parameters during training
function (c::Chain2)(x,y)
    loss = nll(c(x),y)
    if training() # Only apply regularization during training, only to weights, not biases.
        c.λ1 != 0 && (loss += c.λ1 * sum(sum(abs, l.w) for l in c.layers))
        c.λ2 != 0 && (loss += c.λ2 * sum(sum(abs2,l.w) for l in c.layers))
    end
    return loss
end




# Here is a single adam update:
function adamGSupdate!(func, args1, args2, args3, v, G, t; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    t=t+1
    fval1 = @diff func(args1...)
    fval2 = @diff func(args2...)
    fval3 = @diff func(args3...)
    for (w1,w2,w3) in zip(params(fval1),params(fval2),params(fval3))
        g1 = grad(fval1, w1)
        g2 = grad(fval1, w2)
        g3 = grad(fval1, w3)        

        MaskGS = abs.(sign.(g1).+sign.(g2).+sign.(g3)).==3
        
        g = (g1.+g2.+g3).*MaskGS

        v = beta1 * v + (1 - beta1) * g
        G = beta2 * G + (1 - beta2) * g .^ 2
        vhat = v ./ (1 - beta1 ^ t)
        Ghat = G ./ (1 - beta2 ^ t)
        w1 = w1 - (lr / (sqrt(Ghat) + eps)) * vhat
    end
    return value(fval1)
end


# Adam optimizer
function adamGS(func, data; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    v=0;
    G=0;
    t=0;
    (adamGSupdate!(func, args1, args2, args3, v, G, t; lr=lr, beta1=beta1, beta2=beta2, eps=eps) for args in data)
end


# Here is a single adam update:
function adam2update!(func, args1, args2, args3, v, G, t; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    t=t+1
    fval = @diff func(args...)
    for w in params(fval)
        g = grad(fval, w)
        v = beta1 * v + (1 - beta1) * g
        G = beta2 * G + (1 - beta2) * g .^ 2
        vhat = v ./ (1 - beta1 ^ t)
        Ghat = G ./ (1 - beta2 ^ t)
        w = w - (lr / (sqrt(Ghat) + eps)) * vhat
    end
    return value(fval)
end

# Adam optimizer
function adam2(func, data; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    v=0;
    G=0;
    t=0;
    (adam2update!(func, args1, args2, args3, v, G, t; lr=lr, beta1=beta1, beta2=beta2, eps=eps) for args in data)
end






# Definition of SGD update for Gradient Surgery:
function sgdupdateGS!(func, args1, args2, args3; lr=0.1)
    fval1 = @diff func(args1...)
    fval2 = @diff func(args2...)
    fval3 = @diff func(args3...)
    for (param1,param2,param3) in zip(params(fval1),params(fval2),params(fval3))
        ∇param1 = grad(fval1, param1)
        ∇param2 = grad(fval2, param2)
        ∇param3 = grad(fval3, param3)
        
        MaskGS = abs.(sign.(∇param1).+sign.(∇param2).+sign.(∇param3)).==3
        
        param1 .-= lr * ((∇param1.+∇param2.+∇param3).*MaskGS)

    end
    return value(fval1)
end

sgdGS(func, data1, data2, data3; lr=0.1) = 
    (sgdupdateGS!(func, args1, args2, args3; lr=lr) for (args1, args2, args3) in zip(data1, data2, data3))
    


    
    

"""
# Definition of SGD update for Gradient Surgery:
function sgdupdateGS!(func, args1, args2, args3; lr=1e-5)
    fval1 = @diff func(args1...)
    fval2 = @diff func(args2...)
    fval3 = @diff func(args3...)
    for (param1,param2,param3) in zip(params(fval1),params(fval2),params(fval3))
        ∇param1 = grad(fval1, param1)
        ∇param2 = grad(fval2, param2)
        ∇param3 = grad(fval3, param3)
        
        MaskGS = abs.(sign.(∇param1).+sign.(∇param2).+sign.(∇param3)).==3
        
        param1 .-= lr * ((∇param1.+∇param2.+∇param3).*MaskGS)

    end
    return value(fval1)
end

sgdGS(func, data1, data2, data3; lr=1e-5) = 
    (sgdupdateGS!(func, args1, args2, args3; lr=lr) for (args1, args2, args3) in zip(data1, data2, data3))
    
    
    
    
    # Definition of Adam update for Gradient Surgery:
function adamupdateGS!(func, args1, args2, args3; lr=0.001, beta1= 0.9, beta2= 0.999)

    fval1 = @diff func(args1...)
    fval2 = @diff func(args2...)
    fval3 = @diff func(args3...)
    for (param1,param2,param3) in zip(params(fval1),params(fval2),params(fval3))
        ∇param1 = grad(fval1, param1)
        ∇param2 = grad(fval2, param2)
        ∇param3 = grad(fval3, param3)
        
        MaskGS = abs.(sign.(∇param1).+sign.(∇param2).+sign.(∇param3)).==3
        
        
        param1 .-= lr * ((∇param1.+∇param2.+∇param3).*MaskGS)



v = beta1 * v + (1 - beta1) * g
G = beta2 * G + (1 - beta2) * g .^ 2
vhat = v ./ (1 - beta1 ^ t)
Ghat = G ./ (1 - beta2 ^ t)
w = w - (lr / (sqrt(Ghat) + eps)) * vhat

    end
    return value(fval1)
end




# Here is a single adam update:
function adamupdate!(func, args, t; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    t=t+1
    fval = @diff func(args...)
    for w in params(fval)
        g = grad(fval, w)
        v = beta1 * v + (1 - beta1) * g
        G = beta2 * G + (1 - beta2) * g .^ 2
        vhat = v ./ (1 - beta1 ^ t)
        Ghat = G ./ (1 - beta2 ^ t)
        w = w - (lr / (sqrt(Ghat) + eps)) * vhat
    end
    return value(fval)
end

# Adam optimizer
function adam(func, data; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    v=0;
    G=0;
    t=0;
    adamupdate!(func, args, t; lr=lr, beta1=beta1, beta2=beta2, eps=eps) for args in data
end


v = beta1 * v + (1 - beta1) * g
G = beta2 * G + (1 - beta2) * g .^ 2
vhat = v ./ (1 - beta1 ^ t)
Ghat = G ./ (1 - beta2 ^ t)
w = w - (lr / (sqrt(Ghat) + eps)) * vhat



"""