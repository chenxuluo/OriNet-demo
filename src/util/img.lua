function applyFn(fn, t, t2)
    -- Apply an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:int():add(1)
end

function transformPreds(coords, center, scale, res)
    local origDims = coords:size()
    coords = coords:view(-1,2)
    local newCoords = coords:clone()
    for i = 1,coords:size(1) do
        newCoords[i] = transform(coords[i], center, scale, 0, res, 1)
    end
    return newCoords:view(origDims)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function checkDims(dims)
    return dims[3] < dims[4] and dims[5] < dims[6]
end

function crop(img, center, scale, rot, res)
    local ndim = img:nDimension()
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local ht,wd = img:size(2), img:size(3)
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res
    if scaleFactor < 2 then scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
           -- Zoomed out so much that the image is now a single pixel or less
           if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
           return newImg
        else
           tmpImg = image.scale(img,newSize)
           ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end
  

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over
    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       print("Error occurred during crop!")
    end

    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
    return newImg
end



-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------

function localMaxes(hm, n, c, s, hmIdx, nmsWindowSize)
    -- Set up max network for NMS
    local nmsWindowSize = nmsWindowSize or 3
    local nmsPad = (nmsWindowSize - 1)/2
    local maxlayer = nn.Sequential()
    if cudnn then
        maxlayer:add(cudnn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad, nmsPad))
        maxlayer:cuda()
    else
        maxlayer:add(nn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad,nmsPad))
        maxlayer:float()
    end
    maxlayer:evaluate()

    local hmSize = torch.totable(hm:size())
    hm = torch.Tensor(1,unpack(hmSize)):copy(hm):float()
    if hmIdx then hm = hm:sub(1,-1,hmIdx,hmIdx) end
    local hmDim = hm:size()
    local max_out
    -- First do nms
    if cudnn then
        max_out = maxlayer:forward(hm:cuda())
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end

    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    local predCoords = torch.Tensor(hmDim[2], n, 2)
    local predScores = torch.Tensor(hmDim[2], n)
    for i = 1, hmDim[2] do
        local nms_flat = nms[i]:view(nms[i]:nElement())
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            local pt = {(idxs[j]-1) % hmSize[3] + 1, math.floor((idxs[j]-1) / hmSize[3]) + 1 }
            if c then
                predCoords[i][j] = transform(pt, c, s, 0, hmSize[#hmSize], true)
            else
                predCoords[i][j] = torch.Tensor(pt)
            end
            predScores[i][j] = vals[j]
        end
    end
    return predCoords, predScores
end


function heatmapVisualization(set, idx, pred, inp, gt)
    local set = set or 'valid'
    local hmImg
    local tmpInp,tmpHm
    if not inp then
        inp, gt = loadData(set,{idx})
        inp = inp[1]
        gt = gt[1][1]
        tmpInp,tmpHm = inp,gt
    else
        tmpInp = inp
        tmpHm = gt or pred
    end
    local nOut,res = tmpHm:size(1),tmpHm:size(3)
    -- Repeat input image, and darken it to overlay heatmaps
    tmpInp = image.scale(tmpInp,res):mul(.3)
    tmpInp[1][1][1] = 1
    hmImg = tmpInp:repeatTensor(nOut,1,1,1)
    if gt then -- Copy ground truth heatmaps to red channel
        hmImg:sub(1,-1,1,1):add(gt:clone():mul(.7))
    end
    if pred then -- Copy predicted heatmaps to blue channel
        hmImg:sub(1,-1,3,3):add(pred:clone():mul(.7))
    end
    -- Rescale so it is a little easier to see
    hmImg = image.scale(hmImg:view(nOut*3,res,res),256):view(nOut,3,256,256)
    return hmImg, inp
end

