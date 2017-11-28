-- Update dimension references to account for intermediate supervision
ref.predDim = {dataset.nJoints,5}
ref.outputDim = {}

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

-- Code to generate training samples from raw images
function generateSample(idx)
    local img = dataset:loadImage(idx)
    local pts3D_univ, c, s = dataset:getPartInfo(idx)

    local inp = crop(img, c, s, 0, opt.inputRes)
    
    return inp,pts3D_univ
end


function postprocess(set, idx, output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds(tmpOutput)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    -- Transform predictions back to original coordinate space
    --local p_tf = torch.zeros(p:size())
    --for i = 1,p:size(1) do
    --    _,c,s = dataset:getPartInfo(idx[i])
    --    p_tf[i]:copy(transformPreds(p[i], c, s, opt.outputRes))
    --end
    
    --return p_tf:cat(p,3):cat(scores,3)
    return p:cat(scores,3)
end


