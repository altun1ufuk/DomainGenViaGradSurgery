ENV["COLUMNS"]=72
using Base.Iterators: flatten, take, cycle
using IterTools: ncycle, takenth
using Statistics: mean
using CUDA: CUDA, CuArray # functional
using Knet: Knet, conv4, pool, mat, nll, accuracy, progress, sgd, param, param0, dropout, relu, training, minibatch, Data
using Knet, IterTools
using JLD2

# Definition of a convolutional layer (without pooling):
mutable struct Conv1; w; b; f; pd; p; s; end
(c1::Conv1)(x) = c1.f.(conv4(c1.w, dropout(x,c1.pd); padding=c1.p, stride=c1.s) .+ c1.b)
Conv1(w1::Int,w2::Int,cx::Int,cy::Int,f=relu; pdrop=0, padding=0, stride=1)= Conv1(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop, padding, stride)

# Definition of a convolutional layer (with pooling):
mutable struct Conv2; w; b; f; pd; p; s; pw; ps end
(c2::Conv2)(x) = c2.f.(pool(conv4(c2.w, dropout(x,c2.pd); padding=c2.p, stride=c2.s) .+ c2.b; window=c2.pw, stride=c2.ps))
Conv2(w1::Int,w2::Int,cx::Int,cy::Int,f=relu; pdrop=0, padding=0, stride=1, pwindow=3, pstride=2)= Conv2(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop, padding, stride, pwindow, pstride)

# Define a convolutional layer:
mutable struct Conv; w; b; f; p; pad_conv; stri_conv; pad_pool; stri_pool; wind; end
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
mutable struct Dense; w; b; f; p; end
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


function generate_alexnet_model(; pretrained = true)

    L1 = Conv2(11, 11, 3, 64; padding=2, stride=4);
    L2 = Conv2( 5,  5, 64, 192; padding=2);
    L3 = Conv1( 3,  3,  192, 384; padding=1);
    L4 = Conv1( 3,  3,  384, 256; padding=1);
    L5 = Conv2( 3,  3,  256, 256; padding=1);
    L6 = Dense(256*6*6, 4096,pdrop=0.5);
    L7 = Dense(4096, 4096, pdrop=0.5);
    L8 = Dense(4096, 7, identity); 

    if pretrained==true
        state_dict = torch.load("/Users/ufukaltun/Downloads/alexnet-owt-7be5be79.pth");

        L1_w = permutedims(state_dict["features.0.weight"].detach().numpy(), (4, 3, 2, 1));
        L1_b = state_dict["features.0.bias"].detach().numpy();

        L2_w = permutedims(state_dict["features.3.weight"].detach().numpy(), (4, 3, 2, 1));
        L2_b = state_dict["features.3.bias"].detach().numpy();

        L3_w = permutedims(state_dict["features.6.weight"].detach().numpy(), (4, 3, 2, 1));
        L3_b = state_dict["features.6.bias"].detach().numpy();

        L4_w = permutedims(state_dict["features.8.weight"].detach().numpy(), (4, 3, 2, 1));
        L4_b = state_dict["features.8.bias"].detach().numpy();

        L5_w = permutedims(state_dict["features.10.weight"].detach().numpy(), (4, 3, 2, 1));
        L5_b = state_dict["features.10.bias"].detach().numpy();

        L6_w = state_dict["classifier.1.weight"].detach().numpy();
        L6_b = state_dict["classifier.1.bias"].detach().numpy();

        L7_w = state_dict["classifier.4.weight"].detach().numpy();
        L7_b = state_dict["classifier.4.bias"].detach().numpy();

        L8_w = state_dict["classifier.6.weight"].detach().numpy()[1:7,:];
        L8_b = state_dict["classifier.6.bias"].detach().numpy()[1:7];

        L1.w = Param(L1_w);  L1.b = Param(reshape(L1_b, (1,1,:,1)));
        L2.w = Param(L2_w);  L2.b = Param(reshape(L2_b, (1,1,:,1)));
        L3.w = Param(L3_w);  L3.b = Param(reshape(L3_b, (1,1,:,1)));
        L4.w = Param(L4_w);  L4.b = Param(reshape(L4_b, (1,1,:,1)));
        L5.w = Param(L5_w);  L5.b = Param(reshape(L5_b, (1,1,:,1)));
        L6.w = Param(L6_w);  L6.b = Param(L6_b);
        L7.w = Param(L7_w);  L7.b = Param(L7_b);
        L8.w = Param(L8_w);  L8.b = Param(L8_b);
    end

    model = Chain2(L1, L2, L3, L4, L5, L6, L7, L8; λ1=5e-5);
    return model
end
