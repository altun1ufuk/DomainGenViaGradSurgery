using Images
using FileIO: load, save
using Random

function read_data(data_path,image_size)
    i=1
    first_r=true
    y=[]
    x=0
    folders=readdir(data_path)
    folders = folders[folders.!=".DS_Store"]
    for class in folders
        directory=string(data_path,"/",class)
        for img in readdir(directory)
            arr_img=load(string(directory,"/",img))
            if size(arr_img) != image_size
              arr_img = imresize(arr_img, image_size);
            end
            arr_img=channelview(arr_img)
            arr_img = permutedims(arr_img, (2, 3, 1))
            if first_r==false
                x=cat(x,arr_img,dims=4)
            else
                x=cat(arr_img,dims=4)
            end
            push!(y,i)
            first_r=false
        end
        i+=1
        end
    println(size(y))
    println(size(x))
    return x,y;
end


struct PACS
    painting_x
    painting_y
    cartoon_x
    cartoon_y
    photo_x
    photo_y
    sketch_x
    sketch_y
    function PACS(data_path::AbstractString, s1, s2)
        directory=string(data_path,"/art_painting")
        painting_x, painting_y = read_data(directory,(s1,s2));
        save("painting_x.jld2","data",painting_x)
        save("painting_y.jld2","data",painting_y)
        println("Domain: art_painting")

        directory=string(data_path,"/cartoon")
        cartoon_x, cartoon_y = read_data(directory,(s1,s2));
        save("cartoon_x.jld2","data",cartoon_x)
        save("cartoon_y.jld2","data",cartoon_y)
        println("Domain: cartoon")

        directory=string(data_path,"/photo")
        photo_x, photo_y = read_data(directory,(s1,s2));
        save("photo_x.jld2","data",photo_x)
        save("photo_y.jld2","data",photo_y)
        println("Domain: photo")

        directory=string(data_path,"/sketch")
        sketch_x, sketch_y = read_data(directory,(s1,s2));
        save("sketch_x.jld2","data",sketch_x)
        save("sketch_y.jld2","data",sketch_y)
        println("Domain: sketch")
        new(painting_x, painting_y, cartoon_x, cartoon_y, photo_x, photo_y, sketch_x, sketch_y)
    end
    function PACS(data_path::AbstractString)
        painting_x=load(string(data_path,"/painting_x.jld2"))["data"];
        painting_y=load(string(data_path,"/painting_y.jld2"))["data"];

        cartoon_x=load(string(data_path,"/cartoon_x.jld2"))["data"];
        cartoon_y=load(string(data_path,"/cartoon_y.jld2"))["data"];

        photo_x=load(string(data_path,"/photo_x.jld2"))["data"];
        photo_y=load(string(data_path,"/photo_y.jld2"))["data"];

        sketch_x=load(string(data_path,"/sketch_x.jld2"))["data"];
        sketch_y=load(string(data_path,"/sketch_y.jld2"))["data"];
        new(painting_x, painting_y, cartoon_x, cartoon_y, photo_x, photo_y, sketch_x, sketch_y)
    end
end

function PACS(data_path::AbstractString, batchsize, atype, target)
    data = PACS(data_path)
    
            ind = randperm(size(data.painting_x,4))
            val_size = Int(floor(size(data.painting_x,4)*0.1))
            test_size = Int(floor(size(data.painting_x,4)*0.2))
            painting_x = data.painting_x[:,:,:,ind];
            painting_y = data.painting_y[ind];
                painting_val_x = painting_x[:,:,:,1:val_size]
                painting_tst_x = painting_x[:,:,:,val_size+1:val_size+test_size]
                painting_trn_x = painting_x[:,:,:,val_size+test_size+1:end]
                    painting_val_y = painting_y[1:val_size]
                    painting_tst_y = painting_y[val_size+1:val_size+test_size]
                    painting_trn_y = painting_y[val_size+test_size+1:end]
            ind = randperm(size(data.cartoon_x,4))
            val_size = Int(floor(size(data.cartoon_x,4)*0.1))
            test_size = Int(floor(size(data.cartoon_x,4)*0.2))
            cartoon_x = data.cartoon_x[:,:,:,ind];
            cartoon_y = data.cartoon_y[ind];
                cartoon_val_x = cartoon_x[:,:,:,1:val_size]
                cartoon_tst_x = cartoon_x[:,:,:,val_size+1:val_size+test_size]
                cartoon_trn_x = cartoon_x[:,:,:,val_size+test_size+1:end]
                    cartoon_val_y = cartoon_y[1:val_size]
                    cartoon_tst_y = cartoon_y[val_size+1:val_size+test_size]
                    cartoon_trn_y = cartoon_y[val_size+test_size+1:end]
            ind = randperm(size(data.photo_x,4))
            val_size = Int(floor(size(data.photo_x,4)*0.1))
            test_size = Int(floor(size(data.photo_x,4)*0.2))
            photo_x = data.photo_x[:,:,:,ind];
            photo_y = data.photo_y[ind];
                photo_val_x = photo_x[:,:,:,1:val_size]
                photo_tst_x = photo_x[:,:,:,val_size+1:val_size+test_size]
                photo_trn_x = photo_x[:,:,:,val_size+test_size+1:end]
                    photo_val_y = photo_y[1:val_size]
                    photo_tst_y = photo_y[val_size+1:val_size+test_size]
                    photo_trn_y = photo_y[val_size+test_size+1:end]
            ind = randperm(size(data.sketch_x,4))
            val_size = Int(floor(size(data.sketch_x,4)*0.1))
            test_size = Int(floor(size(data.sketch_x,4)*0.2))
            sketch_x = data.sketch_x[:,:,:,ind];
            sketch_y = data.sketch_y[ind];
                sketch_val_x = sketch_x[:,:,:,1:val_size]
                sketch_tst_x = sketch_x[:,:,:,val_size+1:val_size+test_size]
                sketch_trn_x = sketch_x[:,:,:,val_size+test_size+1:end]
                    sketch_val_y = sketch_y[1:val_size]
                    sketch_tst_y = sketch_y[val_size+1:val_size+test_size]
                    sketch_trn_y = sketch_y[val_size+test_size+1:end]
                    
    if target == "sketch"
        d_trn = Knet.minibatch(cat(painting_trn_x, cartoon_trn_x, photo_trn_x, dims=4), cat(painting_trn_y, cartoon_trn_y, photo_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(painting_val_x, cartoon_val_x, photo_val_x, dims=4), cat(painting_val_y, cartoon_val_y, photo_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(sketch_tst_x,sketch_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(photo_tst_x,photo_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(cartoon_tst_x,cartoon_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(painting_tst_x,painting_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        
    elseif target == "photo"
        d_trn = Knet.minibatch(cat(painting_trn_x, cartoon_trn_x, sketch_trn_x, dims=4), cat(painting_trn_y, cartoon_trn_y, sketch_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(painting_val_x, cartoon_val_x, sketch_val_x, dims=4), cat(painting_val_y, cartoon_val_y, sketch_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(photo_tst_x,photo_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(sketch_tst_x,sketch_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(cartoon_tst_x,cartoon_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(painting_tst_x,painting_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        
    elseif target == "cartoon"
        d_trn = Knet.minibatch(cat(painting_trn_x, photo_trn_x, sketch_trn_x, dims=4), cat(painting_trn_y, photo_trn_y, sketch_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(painting_val_x, photo_val_x, sketch_val_x, dims=4), cat(painting_val_y, photo_val_y, sketch_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(cartoon_tst_x,cartoon_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(sketch_tst_x,sketch_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(photo_tst_x,photo_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(painting_tst_x,painting_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        
    elseif target == "painting"
        d_trn = Knet.minibatch(cat(cartoon_trn_x, photo_trn_x, sketch_trn_x, dims=4), cat(cartoon_trn_y, photo_trn_y, sketch_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(cartoon_val_x, photo_val_x, sketch_val_x, dims=4), cat(cartoon_val_y, photo_val_y, sketch_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(painting_tst_x,painting_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(sketch_tst_x,sketch_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(photo_tst_x,photo_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(cartoon_tst_x,cartoon_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
    else
        println("No valid target")
    end
    
    return d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3
end
