# Define a convolutional layer:
struct Conv; w; b; f; p; pad_conv; stri_conv; pad_pool; stri_pool; wind; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p); padding=c.pad_conv, stride=c.stri_conv) .+ c.b ;
        window=c.wind, padding=c.pad_pool, stride=c.stri_pool )) 
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0,pad_conv=(0,0),stri_conv=0,pad_pool=(0,0),stri_pool=0, wind=3)= Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop ,pad_conv,stri_conv,pad_pool,stri_pool, wind)

# Redefine dense layer (See mlp.ipynb):
struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

# Let's define a chain of layers
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
#(c::Chain)(x1,y1,x2,y2,x3,y3) = mean.(nll(c(x1),y1), nll(c(x2),y2), nll(c(x3),y3)) 
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)
#(c::Chain)(d1::Data, d2::Data, d3::Data) = mean( mean(c(x1,y1), c(x2,y2), c(x3,y3)) for (x1,y1,x2,y2,x3,y3) in d1)



# Here is a single SGD update:
function sgdupdateGS!(func, args1, args2, args3; lr=0.1)
    fval1 = @diff func(args1...)
    fval2 = @diff func(args2...)
    fval3 = @diff func(args3...)
    for (param1,param2,param3) in zip(params(fval1),params(fval2),params(fval3))
        ∇param1 = grad(fval1, param1)
        ∇param2 = grad(fval2, param2)
        ∇param3 = grad(fval3, param3)
        
        MaskGS = abs.(sign.(∇param1).+sign.(∇param2).+sign.(∇param3)).==3
        
        param1 .-= lr * ((∇param1.+∇param2.+∇param3).*MaskGS)./ 3
    end
    return value(fval1)
end

sgdGS(func, data1, data2, data3; lr=0.1) = 
    (sgdupdateGS!(func, args1, args2, args3; lr=lr) for (args1, args2, args3) in zip(data1, data2, data3))



"""
# Here is a single SGD update:
function sgdtestupdate!(func, args; lr=0.1)
    fval = @diff func(args...)
    @show size.(args)
    counter = 0
    for param in params(fval)
        ∇param = grad(fval, param)
        param .-= lr * ∇param
        @show size(∇param)
        @show size(param)
        counter += 1
    end
    @show counter
    return value(fval)
end

# We define SGD for a dataset as an iterator so that:
# 1. We can monitor and report the training loss
# 2. We can take snapshots of the model during training
# 3. We can pause/terminate training when necessary
sgdtest(func, data; lr=0.1) = 
    (sgdtestupdate!(func, args; lr=lr) for args in data)
"""




"""
# Here is a single SGD update:
function sgdupdateGS1!(func1, func2, func3, args1, args2, args3; lr=0.1)
    fval1 = @diff func1(args1...)
    fval2 = @diff func2(args2...)
    fval3 = @diff func3(args3...)
    for (param1,param2,param3) in zip(params(fval1),params(fval2),params(fval3))
        ∇param1 = grad(fval1, param1)
        ∇param2 = grad(fval2, param2)
        ∇param3 = grad(fval3, param3)
        
        MaskGS = abs.(sign.(∇param1).+sign.(∇param2).+sign.(∇param3)).==3
        
        param1 .-= lr * (∇param1.*MaskGS)
        param2 .-= lr * (∇param2.*MaskGS)
        param3 .-= lr * (∇param3.*MaskGS)
    end
    @show value(fval1)
    @show value(fval2)
    return value(fval1), value(fval2), value(fval3)
end

sgdGS1(func1, func2, func3, data1, data2, data3; lr=0.1) = 
    (sgdupdateGS1!(func1, func2, func3, args1, args2, args3; lr=lr) for (args1, args2, args3) in zip(data1, data2, data3))
"""




