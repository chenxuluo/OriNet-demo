require 'paths'
require 'torch'
require 'xlua'
paths.dofile('ref.lua')
model = torch.load(opt.loadModel)


file = io.open('Result.txt','a')

ntest = opt.nValidImgs

model:evaluate()
for idx = 1,ntest do
    xlua.progress(idx, ntest)
    inp,pts3D = generateSample(idx)

    pred = model:forward(inp:cuda())
    orien = pred[9][{1,{},{},{}}]:float():clone()
    pred2D = getPreds(pred[10]:float():clone())
    pred2D = pred2D[{1,{},{}}]
    res = 64
    skel3D = torch.zeros(17,3)

    for id = 1,#dataset.skeletonRef do    
        idx1 = dataset.skeletonRef[id][1]
        idx2 = dataset.skeletonRef[id][2]
    
        jnt1 = pred2D[idx1]
        jnt2 = pred2D[idx2]
    
        length = torch.norm(pts3D[idx1]-pts3D[idx2])
        l = torch.norm(jnt2 - jnt1)
        v = (jnt2 - jnt1) / l
        vp = torch.Tensor({v[2],-v[1]})
        tmp = torch.zeros(3)
        left = torch.cmin(jnt1,jnt2)
        right = torch.cmax(jnt1,jnt2)
        for i = left[2],right[2] do
            for j = left[1],right[1] do
                pos = torch.Tensor({j-jnt1[1],i-jnt1[2]})
                pro1 = v:dot(pos)
                pro2 = vp:dot(pos)
                tnorm = torch.norm(orien[{{3*id-2,3*id},i,j}]) 
                if pro1 >= 0 and pro1 <= l and torch.abs(pro2) <= 1 and tnorm > 0.1 then
                    tmp = tmp + orien[{{3*id-2,3*id},i,j}] / tnorm
                end
            end
        end
        if torch.norm(tmp) < 0.1 then
            for ii = 1,3 do
                t1 = torch.max(orien[{id*3-3+ii,{},{}}])
                t2 = torch.min(orien[{id*3-3+ii,{},{}}])
                if t1 + t2 < 0 then
            	   tmp[ii] = t2
                else
            	   tmp[ii] = t1
                end
            end
        end
        skel3D[idx2] = skel3D[idx1]:clone() + tmp:clone() / torch.norm(tmp) * length 

    end
        
    skel3D = skel3D:double()
    tmp = pts3D[{1,{}}]:clone()
   
    for i = 1,17 do
        pts3D[{i,{}}] = pts3D[{i,{}}]:clone() - tmp:clone()
    end
  
    diss = torch.sqrt(torch.sum(torch.pow(pts3D:clone()-skel3D:clone(),2),2)):transpose(1,2):squeeze()
    
    for i=1,17 do
        file:write(diss[i] .. '\t')
    end
    file:write('\n')
    file:flush()
end

