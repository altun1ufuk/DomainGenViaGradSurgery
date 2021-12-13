using Images
using FileIO: load, save

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
    painting=Knet.minibatch(data.painting_x,data.painting_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
    cartoon=Knet.minibatch(data.cartoon_x,data.cartoon_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
    photo=Knet.minibatch(data.photo_x,data.photo_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
    sketch=Knet.minibatch(data.sketch_x,data.sketch_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
    
    if target == "sketch"
        baseline = Knet.minibatch(cat(data.painting_x, data.cartoon_x, data.photo_x, dims=4), cat(data.painting_y, data.cartoon_y, data.photo_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
    elseif target == "photo"
        baseline = Knet.minibatch(cat(data.painting_x, data.cartoon_x, data.sketch_x, dims=4), cat(data.painting_y, data.cartoon_y, data.sketch_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
    elseif target == "cartoon"
        baseline = Knet.minibatch(cat(data.painting_x, data.photo_x, data.sketch_x, dims=4), cat(data.painting_y, data.photo_y, data.sketch_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});        
    elseif target == "painting"
        baseline = Knet.minibatch(cat(data.cartoon_x, data.photo_x, data.sketch_x, dims=4), cat(data.cartoon_y, data.photo_y, data.sketch_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32}); 
    else
        println("No valid target")
    end
    return painting, cartoon, photo, sketch, baseline
end
