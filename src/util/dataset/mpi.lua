local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 17
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
    self.flipRef = {{2,5},   {3,6},   {4,7},
                    {12,15}, {13,16}, {14,17}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},   {3,4,1},
                        {1,5,2},    {5,6,2},   {6,7,2},
                        {1,8,0},    {8,9,0},   {9,10,0}, {10,11,0},
                        {9,15,3},   {15,16,3}, {16,17,3},
                        {9,12,4},   {12,13,4}, {13,14,4}} 

    local annot = {}
    local tags = {'index','imagename','center','scale','part_3D_univ'} -- 'part_3D' (global world coordinate)
    local a = hdf5.open(paths.concat(projectDir,'data/mpi/annot-test.h5'),'r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()

    opt.nValidImgs = annot.index:size(1)
    self.annot = annot
end

function Dataset:getPath(idx)
    return paths.concat(opt.dataDir,ffi.string(self.annot.imagename[idx]:char():data()))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts3D_univ = self.annot.part_3D_univ[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    return pts3D_univ, c, s
end

return M.Dataset

