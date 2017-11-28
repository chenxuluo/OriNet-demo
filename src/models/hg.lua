paths.dofile('Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
   
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end


function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,128)(r4)
    local r6 = Residual(128,opt.nFeats)(r5)
    local out = {}
    local inter = r6

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll1 = Residual(opt.nFeats,opt.nFeats)(ll) end
        for j = 1,opt.nModules do ll2 = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll1 = lin(opt.nFeats,opt.nFeats,ll1)
        ll2 = lin(opt.nFeats,opt.nFeats,ll2)
        -- Predicted heatmaps
        local tmpOut1 = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(ll1)
        local tmpOut2 = nnlib.Sigmoid()(nnlib.SpatialConvolution(opt.nFeats, ref.nJoints,1,1,1,1,0,0)(ll2))
	    table.insert(out,tmpOut1)
	    table.insert(out,tmpOut2)
       

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut1_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(tmpOut1)
            local tmpOut2_ = nnlib.SpatialConvolution(ref.nJoints,opt.nFeats,1,1,1,1,0,0)(tmpOut2)
	        inter = nn.CAddTable()({r6, ll_,tmpOut1_,tmpOut2_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)
    
   return model

end
