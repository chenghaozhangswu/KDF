// bench_exhaustive.cpp - 全面对比: dim(10/20/30) x top(10/50/200/500) x 双树
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <algorithm>
#include <random>
#include <immintrin.h>
#include <nanoflann.hpp>
using namespace std;
using f32 = float;
constexpr int N=601,NMAT=4,NQ=500;
const char*MNP[]={"ox","sin","soi","cauthy"};
const int SZV[]={10000,50000,100000,500000};
vector<f32>loadf(const string&p){
    FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"FAIL: %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END);size_t sz=ftell(f);fseek(f,0,SEEK_SET);
    vector<f32>b(sz/4);fread(b.data(),4,b.size(),f);fclose(f);return b;
}
inline f32 d2(const f32*a,const f32*b,int n){
    __m256 s=_mm256_setzero_ps();int k=0;
    for(;k+8<=n;k+=8){__m256 va=_mm256_loadu_ps(a+k),vb=_mm256_loadu_ps(b+k),d=_mm256_sub_ps(va,vb);
    s=_mm256_add_ps(s,_mm256_mul_ps(d,d));}
    __m128 lo=_mm256_castps256_ps128(s),hi=_mm256_extractf128_ps(s,1);
    __m128 ss=_mm_add_ps(lo,hi);ss=_mm_hadd_ps(ss,ss);ss=_mm_hadd_ps(ss,ss);
    f32 r=_mm_cvtss_f32(ss);for(;k<n;k++){f32 d=a[k]-b[k];r+=d*d;}return r;
}

template<int D>
struct PC{vector<f32>pts;size_t kdtree_get_point_count()const{return pts.size()/D;}
    f32 kdtree_get_pt(size_t i,size_t d)const{return pts[i*(size_t)D+d];}
    template<class B>bool kdtree_get_bbox(B&)const{return false;}};
template<int D> using KT=nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<f32,PC<D>>,PC<D>,D>;

void pcproj(const f32*r,f32*o,const f32*m,const f32*c,int nd){
    for(int d=0;d<nd;d++){const f32*cp=c+d*N;__m256 s=_mm256_setzero_ps();
    for(int j=0;j<=592;j+=8){__m256 rd=_mm256_sub_ps(_mm256_loadu_ps(r+j),_mm256_loadu_ps(m+j));
    s=_mm256_fmadd_ps(rd,_mm256_loadu_ps(cp+j),s);}
    f32 bf[8];_mm256_storeu_ps(bf,s);f32 t=bf[0]+bf[1]+bf[2]+bf[3]+bf[4]+bf[5]+bf[6]+bf[7];
    for(int j=600;j<N;j++)t+=(r[j]-m[j])*cp[j];o[d]=t;}
}

f32 var_ratio(const f32*S,int nkeep,int total){
    f32 sk=0,sa=0;for(int i=0;i<total;i++){if(i<nkeep)sk+=S[i];sa+=S[i];}return sk/sa;
}

int main(){
    setbuf(stdout,NULL);
    string BP="D:\\kd_forest_v2_gh\\src\\multi\\";
    printf("=== Exhaustive benchmark: dim(10/20/30) x top(10/50/200/500) x dual-tree ===\n");
    auto gm10=loadf(BP+"gmean_10k.bin");auto gc50=loadf(BP+"gcomp50_10k.bin");
    // extract 10D,20D,30D components
    for(int si=0;si<4;si++){
        int NR=SZV[si],NTOT=NR*NMAT;string SN=si==0?"10k":si==1?"50k":si==2?"100k":"500k";
        printf("\n=== %s (NR=%d) ===\n",SN.c_str(),NR);
        // Load per-scale PCA
        auto gm=loadf(BP+"pca_mean_"+SN+".bin");auto gc50=loadf(BP+"pca_comp50_"+SN+".bin");
        vector<f32>gc10(10*N),gc20(20*N),gc30(30*N);
        for(int d=0;d<10;d++)memcpy(&gc10[d*N],&gc50[d*N],N*4);
        for(int d=0;d<20;d++)memcpy(&gc20[d*N],&gc50[d*N],N*4);
        for(int d=0;d<30;d++)memcpy(&gc30[d*N],&gc50[d*N],N*4);
        vector<vector<f32>>ms(NMAT),mt(NMAT);
        for(int m=0;m<NMAT;m++){ms[m]=loadf(BP+"lib_"+MNP[m]+"_n_"+SN+".bin");
            mt[m]=loadf(BP+"lib_"+MNP[m]+"_thick_"+SN+".bin");}
        vector<f32>as(NTOT*N),at(NTOT);
        for(int m=0;m<NMAT;m++){memcpy(&as[m*NR*N],ms[m].data(),NR*N*4);memcpy(&at[m*NR],mt[m].data(),NR*4);}
        // projections
        vector<f32>p10(NTOT*10),p20(NTOT*20),p30(NTOT*30);
        for(int i=0;i<NTOT;i++){pcproj(&as[i*N],&p10[i*10],gm.data(),gc10.data(),10);
            pcproj(&as[i*N],&p20[i*20],gm.data(),gc20.data(),20);
            pcproj(&as[i*N],&p30[i*30],gm.data(),gc30.data(),30);}
        // trees
        PC<10> pc10d{p10};KT<10> kt10(10,pc10d,nanoflann::KDTreeSingleIndexAdaptorParams(30));kt10.buildIndex();
        PC<20> pc20d{p20};KT<20> kt20(20,pc20d,nanoflann::KDTreeSingleIndexAdaptorParams(30));kt20.buildIndex();
        PC<30> pc30d{p30};KT<30> kt30(30,pc30d,nanoflann::KDTreeSingleIndexAdaptorParams(30));kt30.buildIndex();
        // queries
        mt19937 rng(42);uniform_int_distribution<int>ui(0,NR-1);int NQT=NMAT*NQ;
        vector<f32>qc(NQT*N),qt(NQT);
        for(int m=0;m<NMAT;m++)for(int i=0;i<NQ;i++){int idx=ui(rng);
            memcpy(&qc[(m*NQ+i)*N],&ms[m][idx*N],N*4);qt[m*NQ+i]=mt[m][idx];}
        // noise: 0.5%
        for(int ni=1;ni<2;ni++){ // just 0.5% noise for brevity
            f32 nlev=0.005f;
            printf("  0.5%% noise:\n");
            vector<f32>qn(NQT*N);mt19937 ng(123);normal_distribution<f32>nd2(0,1);
            for(int i=0;i<NQT*N;i++){f32 v=qc[i]+nlev*nd2(ng);if(v<0)v=0;if(v>1)v=1;qn[i]=v;}
            // BF baseline
            auto t1=chrono::high_resolution_clock::now();f32 bf_p1=0;
            for(int qi=0;qi<NQT;qi++){int m=qi/NQ;f32 bd=1e30f,bt=0;
                for(int i=0;i<NR;i++){f32 d=d2(&qn[qi*N],&ms[m][i*N],N);if(d<bd){bd=d;bt=mt[m][i];}}
                if(fabs(bt-qt[qi])<=1)bf_p1++;}
            double bf_lat=chrono::duration<double,micro>(chrono::high_resolution_clock::now()-t1).count()/NQT;
            printf("    BF: P1=%.1f%% lat=%.0fus\n",100*bf_p1/NQT,bf_lat);

            // Test all (dim x topN) combos
            struct Config{int dim;int top;f32*comp;f32*proj;void*tree;};
            Config cfgs[]={
                {10,10,gc10.data(),p10.data(),&kt10},{10,50,gc10.data(),p10.data(),&kt10},
                {10,200,gc10.data(),p10.data(),&kt10},{10,500,gc10.data(),p10.data(),&kt10},
                {20,10,gc20.data(),p20.data(),&kt20},{20,50,gc20.data(),p20.data(),&kt20},
                {20,200,gc20.data(),p20.data(),&kt20},{20,500,gc20.data(),p20.data(),&kt20},
                {30,10,gc30.data(),p30.data(),&kt30},{30,50,gc30.data(),p30.data(),&kt30},
                {30,200,gc30.data(),p30.data(),&kt30},{30,500,gc30.data(),p30.data(),&kt30},
            };
            for(auto&cfg:cfgs){
                int D=cfg.dim,TOP=cfg.top;
                t1=chrono::high_resolution_clock::now();f32 cf_p1=0;
                for(int qi=0;qi<NQT;qi++){
                    f32 qp[30];pcproj(&qn[qi*N],qp,gm.data(),cfg.comp,D);
                    vector<size_t>ri(TOP);vector<f32>rd(TOP);
                    nanoflann::KNNResultSet<f32>rs(TOP);rs.init(&ri[0],&rd[0]);
                    if(D==10)((KT<10>*)cfg.tree)->findNeighbors(rs,qp,nanoflann::SearchParameters());
                    else if(D==20)((KT<20>*)cfg.tree)->findNeighbors(rs,qp,nanoflann::SearchParameters());
                    else ((KT<30>*)cfg.tree)->findNeighbors(rs,qp,nanoflann::SearchParameters());
                    f32 bd=1e30f,bt=0;
                    for(int k=0;k<TOP;k++){int idx=(int)ri[k];f32 d2v=d2(&qn[qi*N],&as[idx*N],N);if(d2v<bd){bd=d2v;bt=at[idx];}}
                    if(fabs(bt-qt[qi])<=1)cf_p1++;}
                double lat=chrono::duration<double,micro>(chrono::high_resolution_clock::now()-t1).count()/NQT;
                printf("    %2dD+Top%3d: P1=%.1f%% lat=%.0fus\n",D,TOP,100*cf_p1/NQT,lat);
            }
            // Dual tree: 10D+20D candidates merge
            t1=chrono::high_resolution_clock::now();f32 cf_p1=0;
            for(int qi=0;qi<NQT;qi++){
                f32 q10[10],q20[20];pcproj(&qn[qi*N],q10,gm10.data(),gc10.data(),10);
                pcproj(&qn[qi*N],q20,gm10.data(),gc20.data(),20);
                int TOP=50;
                vector<size_t>ri1(TOP),ri2(TOP);vector<f32>rd1(TOP),rd2(TOP);
                nanoflann::KNNResultSet<f32>rs1(TOP),rs2(TOP);rs1.init(&ri1[0],&rd1[0]);rs2.init(&ri2[0],&rd2[0]);
                kt10.findNeighbors(rs1,q10,nanoflann::SearchParameters());
                kt20.findNeighbors(rs2,q20,nanoflann::SearchParameters());
                // merge candidates
                vector<int>cand;cand.reserve(TOP*2);
                for(int k=0;k<TOP;k++){cand.push_back((int)ri1[k]);cand.push_back((int)ri2[k]);}
                sort(cand.begin(),cand.end());
                cand.erase(unique(cand.begin(),cand.end()),cand.end());
                f32 bd=1e30f,bt=0;
                for(int idx:cand){f32 d2v=d2(&qn[qi*N],&as[idx*N],N);if(d2v<bd){bd=d2v;bt=at[idx];}}
                if(fabs(bt-qt[qi])<=1)cf_p1++;}
            double lat=chrono::duration<double,micro>(chrono::high_resolution_clock::now()-t1).count()/NQT;
            printf("    10D+20D+50: P1=%.1f%% lat=%.0fus (merge)\n",100*cf_p1/NQT,lat);
        }
    }
    printf("Done\n");return 0;
}