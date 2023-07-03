import math

import torch
from tqdm.notebook import tqdm
from scipy.stats import entropy
use_gpu = torch.cuda.is_available()
import time

# ========================================
#      loading datas

#

def centerDatas(datas):
    datas= datas - datas.mean(1, keepdim=True)
    datas = datas / torch.norm(datas, dim=2, keepdim= True)

    return datas

def scaleEachUnitaryDatas(datas):
  
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1),'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas

def SVDreduction(ndatas,K):
    # ndatas = torch.linear.qr(datas.permute(0, 2, 1),'reduced').R
    # ndatas = ndatas.permute(0, 2, 1)
    _,s,v = torch.svd(ndatas)
    ndatas = ndatas.matmul(v[:,:,:K])

    return ndatas


def predict(gamma, Z, labels):
    # #Certainty_scores = 1 + (Z*torch.log(Z)).sum(dim=2) / math.log(5)
    # Z[:,:n_lsamples].fill_(0)
    # Z[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    #tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
    delta = torch.sum(Z, 1)
    #L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
    iden = torch.eye(5,device='cuda')
    iden = iden.reshape((1, 5, 5))
    iden = iden.repeat(10000, 1, 1)
    W = torch.bmm(torch.transpose(Z,1,2), Z/delta.unsqueeze(1))
    #W = W/W.sum(1).unsqueeze(1)
    #isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W, dim=-1,keepdim=True))
    # checknan(laplacian=isqrt_diag)
    #W = W * isqrt_diag[:, None, :] * isqrt_diag[:, :, None]
    #W = W * isqrt_diag * torch.transpose(isqrt_diag,dim0=2,dim1=1)
    L = iden - W#(W + W.bmm(W))/2
    Z_l = Z[:,:n_lsamples]

    #A = np.dot(np.linalg.inv(torch.matmul(torch.transpose(Z_l,1,2), Z_l) + gamma * L), torch.bmm(torch.transpose(Z_l,1,2), Y))
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*iden)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    Pred = Z.bmm(A)
    normalizer = torch.sum(Pred,dim=1,keepdim=True)
    # #normalizer = Pred[:,:n_lsamples].max(dim=1)[0].unsqueeze(1)
    Pred = (n_shot+n_queries)*Pred/normalizer
    # normalizer = torch.sum(Pred, dim=2, keepdim=True)
    # Pred = Pred/normalizer
    # Pred[:, :n_lsamples].fill_(0)
    # Pred[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
    # N = PredZ.shape[0]
    # K = PredZ.shape[1]
    # pred = np.zeros((N, K))
    #
    # for k in range(K):
    #     current_pred = np.dot(Z, A[:, k])

    return Pred#.clamp(0,1)

def predictW(gamma, Z, labels):
    # #Certainty_scores = 1 + (Z*torch.log(Z)).sum(dim=2) / math.log(5)
    # Z[:,:n_lsamples].fill_(0)
    # Z[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
    delta = torch.sum(Z, 1)
    L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
    Z_l = Z[:,:n_lsamples]

    #A = np.dot(np.linalg.inv(torch.matmul(torch.transpose(Z_l,1,2), Z_l) + gamma * L), torch.bmm(torch.transpose(Z_l,1,2), Y))
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*
    #u = torch.linalg.cholesky(gamma * L)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    P = Z.bmm(A)
    _, n, m = P.shape
    r = torch.ones(n_runs, n_lsamples + n_usamples,device='cuda')
    c = torch.ones(n_runs, n_ways,device='cuda') * (n_shot + n_queries)
    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
    while torch.max(torch.abs(u - P.sum(2))) > 0.01:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        P[:,:n_lsamples].fill_(0)
        P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        if iters == maxiters:
            break
        iters = iters + 1
    return P

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self, ndatas, n_runs, n_shot, n_queries, n_ways, n_nfeat):
        self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:n_shot,].mean(1)
        self.mus = self.mus_ori.clone()
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)
        # self.mus_ori = torch.randn(n_runs, n_ways,n_nfeat,device='cuda')
        # self.mus_ori = self.mus_ori/self.mus_ori.norm(dim=2,keepdim=True)
        # self.mus = self.mus_ori.clone()

    def initFromCenter(self, mus):
        #self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:1,].mean(1)
        self.mus = mus
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)
        # self.mus_ori = torch.randn(n_runs, n_ways,n_nfeat,device='cuda')
        # self.mus_ori = self.mus_ori/self.mus_ori.norm(dim=2,keepdim=True)
        # self.mus = self.mus_ori.clone()

    def updateFromEstimate(self, estimate, alpha, l2 = False):

        diff = self.mus_ori - self.mus
        Dmus = estimate - self.mus
        if l2 == True:
            self.mus = self.mus + alpha * (Dmus) + 0.01 * diff
        else:
            self.mus = self.mus + alpha * (Dmus)
        #self.mus/=self.mus.norm(dim=2, keepdim=True)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
                                         
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, ndatas, n_runs, n_ways, n_usamples, n_lsamples):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        p_xj = torch.zeros_like(dist)
        # r = torch.ones(n_runs, n_usamples)
        # c = torch.ones(n_runs, n_ways) * n_queries
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * (n_queries)
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-3)
        # _, y_pseudo = torch.max(p_xj_test, 2)
        # Certainty_scores = 1 + (p_xj_test*torch.log(p_xj_test)).sum(axis=2) / math.log(5)
        # Certainty_scores = Certainty_scores.unsqueeze(2)
        #p_xj = torch.where(p_xj > 0.9, torch.tensor(1.,device='cuda'), p_xj)
        # p_xj_test[alpha[0],alpha[1],:].fill_(0)
        # p_xj_test[alpha[0],alpha[1],:].scatter_(2, y_pseudo[alpha[0],alpha[1]], 1)
        #sup_alpha = np.where(Certainty_scores >= alpha)[0]
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        
        return p_xj

    def estimateFromMask(self, mask, ndatas):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, epochInfo=None):
     
        p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas,ndatas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)
        #self.alpha -= 0.001
        if self.verbose:
            op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=20):
        
        self.probas = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}".format(epoch, self.alpha))
            p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            self.probas = p_xj

            if self.verbose:
                print("accuracy from filtered probas", self.getAccuracy(self.probas))
            pesudo_L = predictW(0.05, self.probas, labels)
            if self.verbose:
                print("accuracy from AnchorGraph probas", self.getAccuracy(pesudo_L))
            #(pesudo_L + self.probas)
            beta = 0.7
            # p_xj[:,:n_lsamples].fill_(0)
            # p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
            # beta*pesudo_L + (1-beta)*self.probas
            #pesudo_L[:,n_lsamples:] = (beta * pesudo_L[:,n_lsamples:] + (1 - beta) * self.probas[:,n_lsamples:])
            m_estimates = model.estimateFromMask((beta*pesudo_L + (1-beta)*p_xj).clamp(0,1), ndatas)
            #m_estimates = model.estimateFromMask(pesudo_L.clamp(0, 1), ndatas)
            #m_estimates = model.estimateFromMask(p_xj, ndatas)

            # update centroids
            model.updateFromEstimate(m_estimates, self.alpha)
            # self.alpha -= 0.001
            if self.verbose:
                op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
                acc = self.getAccuracy(op_xj)
                print("output model accuracy", acc)
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        acc = self.getAccuracy(op_xj)
        return acc
    

if __name__ == '__main__':
# ---- data loading
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs=10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    #FSLTask.loadDataSet("cross")
    FSLTask.loadDataSet("Res12_miniimagenet")
    #FSLTask.loadDataSet("densenet_tierdimagenet")
    #FSLTask.loadDataSet("Res12_cifar")
    #FSLTask.loadDataSet("cifar")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    _maxRuns = n_runs
    ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shot+n_queries,5).clone().view(n_runs, n_samples)
    
    # Power transform
    beta = 0.5
    #ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta)
    #ndatas = centerDatas(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)
    #ndatas = QRreduction(ndatas)
    #ndatas = Coles(ndatas, 40, 10)
    #ndatas = centerDatas(ndatas)
    ndatas = SVDreduction(ndatas,40)
    n_nfeat = ndatas.size(2)
    #rp = 1./math.sqrt(ndatas.shape[2])*torch.randn((ndatas.shape[2],160))
    #ndatas = ndatas.matmul(rp)

    
    #ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub
     ## very important for QR
    ndatas = centerDatas(ndatas)
    #ndatas = scaleEachUnitaryDatas(ndatas)

    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()
    
    #MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas(ndatas, n_runs, n_shot,n_queries,n_ways,n_nfeat)
    
    alpha = 0.2
    optim = MAP(alpha)

    optim.verbose=True
    optim.progressBar=True
    #for i in range(100):
    T1 = time.perf_counter()
    acc_test = optim.loop(model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=50)
    print('running time:%s ' % (time.perf_counter() - T1))
    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100*x for x in acc_test)))
    
    

