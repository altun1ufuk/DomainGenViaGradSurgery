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
            if size(arr_img) == (3,227,227)
                arr_img = permutedims(arr_img, (2, 3, 1))
                if first_r==false
                    x=cat(x,arr_img,dims=4)
                else
                    x=cat(arr_img,dims=4)
                end
                push!(y,i)
                first_r=false
            end
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




struct VLCS
    Caltech101_x
    Caltech101_y
    LabelMe_x
    LabelMe_y
    SUN09_x
    SUN09_y
    VOC2007_x
    VOC2007_y
    function VLCS(data_path::AbstractString, s1, s2)
        directory=string(data_path,"/Caltech101")
        Caltech101_x, Caltech101_y = read_data(directory,(s1,s2));
        save("Caltech101_x.jld2","data",Caltech101_x)
        save("Caltech101_y.jld2","data",Caltech101_y)
        println("Domain: Caltech101")

        directory=string(data_path,"/LabelMe")
        LabelMe_x, LabelMe_y = read_data(directory,(s1,s2));
        save("LabelMe_x.jld2","data",LabelMe_x)
        save("LabelMe_y.jld2","data",LabelMe_y)
        println("Domain: LabelMe")

        directory=string(data_path,"/SUN09")
        SUN09_x, SUN09_y = read_data(directory,(s1,s2));
        save("SUN09_x.jld2","data",SUN09_x)
        save("SUN09_y.jld2","data",SUN09_y)
        println("Domain: SUN09")

        directory=string(data_path,"/VOC2007")
        VOC2007_x, VOC2007_y = read_data(directory,(s1,s2));
        save("VOC2007_x.jld2","data",VOC2007_x)
        save("VOC2007_y.jld2","data",VOC2007_y)
        println("Domain: VOC2007")
        new(Caltech101_x, Caltech101_y, LabelMe_x, LabelMe_y, SUN09_x, SUN09_y, VOC2007_x, VOC2007_y)
    end
    function VLCS(data_path::AbstractString)
        Caltech101_x=load(string(data_path,"/Caltech101_x.jld2"))["data"];
        Caltech101_y=load(string(data_path,"/Caltech101_y.jld2"))["data"];

        LabelMe_x=load(string(data_path,"/LabelMe_x.jld2"))["data"];
        LabelMe_y=load(string(data_path,"/LabelMe_y.jld2"))["data"];

        SUN09_x=load(string(data_path,"/SUN09_x.jld2"))["data"];
        SUN09_y=load(string(data_path,"/SUN09_y.jld2"))["data"];

        VOC2007_x=load(string(data_path,"/VOC2007_x.jld2"))["data"];
        VOC2007_y=load(string(data_path,"/VOC2007_y.jld2"))["data"];
        new(Caltech101_x, Caltech101_y, LabelMe_x, LabelMe_y, SUN09_x, SUN09_y, VOC2007_x, VOC2007_y)
    end
end

function VLCS(data_path::AbstractString, batchsize, atype, target)
    data = VLCS(data_path)
    
            ind = randperm(size(data.Caltech101_x,4))
            val_size = Int(floor(size(data.Caltech101_x,4)*0.1))
            test_size = Int(floor(size(data.Caltech101_x,4)*0.2))
            Caltech101_x = data.Caltech101_x[:,:,:,ind];
            Caltech101_y = data.Caltech101_y[ind];
                Caltech101_val_x = Caltech101_x[:,:,:,1:val_size]
                Caltech101_tst_x = Caltech101_x[:,:,:,val_size+1:val_size+test_size]
                Caltech101_trn_x = Caltech101_x[:,:,:,val_size+test_size+1:end]
                    Caltech101_val_y = Caltech101_y[1:val_size]
                    Caltech101_tst_y = Caltech101_y[val_size+1:val_size+test_size]
                    Caltech101_trn_y = Caltech101_y[val_size+test_size+1:end]
            ind = randperm(size(data.LabelMe_x,4))
            val_size = Int(floor(size(data.LabelMe_x,4)*0.1))
            test_size = Int(floor(size(data.LabelMe_x,4)*0.2))
            LabelMe_x = data.LabelMe_x[:,:,:,ind];
            LabelMe_y = data.LabelMe_y[ind];
                LabelMe_val_x = LabelMe_x[:,:,:,1:val_size]
                LabelMe_tst_x = LabelMe_x[:,:,:,val_size+1:val_size+test_size]
                LabelMe_trn_x = LabelMe_x[:,:,:,val_size+test_size+1:end]
                    LabelMe_val_y = LabelMe_y[1:val_size]
                    LabelMe_tst_y = LabelMe_y[val_size+1:val_size+test_size]
                    LabelMe_trn_y = LabelMe_y[val_size+test_size+1:end]
            ind = randperm(size(data.SUN09_x,4))
            val_size = Int(floor(size(data.SUN09_x,4)*0.1))
            test_size = Int(floor(size(data.SUN09_x,4)*0.2))
            SUN09_x = data.SUN09_x[:,:,:,ind];
            SUN09_y = data.SUN09_y[ind];
                SUN09_val_x = SUN09_x[:,:,:,1:val_size]
                SUN09_tst_x = SUN09_x[:,:,:,val_size+1:val_size+test_size]
                SUN09_trn_x = SUN09_x[:,:,:,val_size+test_size+1:end]
                    SUN09_val_y = SUN09_y[1:val_size]
                    SUN09_tst_y = SUN09_y[val_size+1:val_size+test_size]
                    SUN09_trn_y = SUN09_y[val_size+test_size+1:end]
            ind = randperm(size(data.VOC2007_x,4))
            val_size = Int(floor(size(data.VOC2007_x,4)*0.1))
            test_size = Int(floor(size(data.VOC2007_x,4)*0.2))
            VOC2007_x = data.VOC2007_x[:,:,:,ind];
            VOC2007_y = data.VOC2007_y[ind];
                VOC2007_val_x = VOC2007_x[:,:,:,1:val_size]
                VOC2007_tst_x = VOC2007_x[:,:,:,val_size+1:val_size+test_size]
                VOC2007_trn_x = VOC2007_x[:,:,:,val_size+test_size+1:end]
                    VOC2007_val_y = VOC2007_y[1:val_size]
                    VOC2007_tst_y = VOC2007_y[val_size+1:val_size+test_size]
                    VOC2007_trn_y = VOC2007_y[val_size+test_size+1:end]
 

    if target == "VOC2007"
        d_trn = Knet.minibatch(cat(Caltech101_trn_x, LabelMe_trn_x, SUN09_trn_x, dims=4), cat(Caltech101_trn_y, LabelMe_trn_y, SUN09_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(Caltech101_val_x, LabelMe_val_x, SUN09_val_x, dims=4), cat(Caltech101_val_y, LabelMe_val_y, SUN09_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(VOC2007_tst_x,VOC2007_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(SUN09_tst_x,SUN09_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(LabelMe_tst_x,LabelMe_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(Caltech101_tst_x,Caltech101_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        
    elseif target == "SUN09"
        d_trn = Knet.minibatch(cat(Caltech101_trn_x, LabelMe_trn_x, VOC2007_trn_x, dims=4), cat(Caltech101_trn_y, LabelMe_trn_y, VOC2007_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(Caltech101_val_x, LabelMe_val_x, VOC2007_val_x, dims=4), cat(Caltech101_val_y, LabelMe_val_y, VOC2007_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(SUN09_tst_x,SUN09_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(VOC2007_tst_x,VOC2007_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(LabelMe_tst_x,LabelMe_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(Caltech101_tst_x,Caltech101_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        
    elseif target == "LabelMe"
        d_trn = Knet.minibatch(cat(Caltech101_trn_x, SUN09_trn_x, VOC2007_trn_x, dims=4), cat(Caltech101_trn_y, SUN09_trn_y, VOC2007_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(Caltech101_val_x, SUN09_val_x, VOC2007_val_x, dims=4), cat(Caltech101_val_y, SUN09_val_y, VOC2007_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(LabelMe_tst_x,LabelMe_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(VOC2007_tst_x,VOC2007_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(SUN09_tst_x,SUN09_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(Caltech101_tst_x,Caltech101_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        
    elseif target == "Caltech101"
        d_trn = Knet.minibatch(cat(LabelMe_trn_x, SUN09_trn_x, VOC2007_trn_x, dims=4), cat(LabelMe_trn_y, SUN09_trn_y, VOC2007_trn_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_val = Knet.minibatch(cat(LabelMe_val_x, SUN09_val_x, VOC2007_val_x, dims=4), cat(LabelMe_val_y, SUN09_val_y, VOC2007_val_y, dims=1), batchsize*3, shuffle=true, xtype=atype, ytype=Array{Int32});
        d_tst = Knet.minibatch(Caltech101_tst_x,Caltech101_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom1 = Knet.minibatch(VOC2007_tst_x,VOC2007_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom2 = Knet.minibatch(SUN09_tst_x,SUN09_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
        d_trn_dom3 = Knet.minibatch(LabelMe_tst_x,LabelMe_tst_y,batchsize,shuffle=true,xtype=atype,ytype=Array{Int32});
    else
        println("No valid target")
    end
    
    return d_trn, d_val, d_tst, d_trn_dom1, d_trn_dom2, d_trn_dom3
end

         